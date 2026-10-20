"""Training losses, optimization loop, and predictor evaluation."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata
from torch.optim.lr_scheduler import ReduceLROnPlateau

from main.runtime import autocast_context, forward_sample_batch, unpack_batch


def pairwise_ranking_loss(
        pred_spread,
        true_spread,
        tie_threshold,
        temperature,
        sample_weights=None,
):
    """Logistic pairwise loss on non-tied spread pairs.

    The former fixed-margin hinge objective could sit near its margin while all
    predicted spreads were almost identical. Temperature-scaled logistic loss
    keeps a useful gradient at equality and explicitly rewards the correct
    ordering direction.
    """
    if pred_spread.numel() < 2:
        return pred_spread.sum() * 0.0

    pred_diff = pred_spread.unsqueeze(1) - pred_spread.unsqueeze(0)
    true_diff = true_spread.unsqueeze(1) - true_spread.unsqueeze(0)
    unique_pairs = torch.triu(
        torch.ones_like(true_diff, dtype=torch.bool), diagonal=1
    )
    mask = unique_pairs & (true_diff.abs() > tie_threshold)
    if not mask.any():
        return pred_spread.sum() * 0.0

    signed_gap = true_diff[mask].sign() * pred_diff[mask]
    pair_losses = F.softplus(-signed_gap / temperature)
    if sample_weights is None:
        return pair_losses.mean()
    pair_weights = torch.maximum(
        sample_weights.unsqueeze(1),
        sample_weights.unsqueeze(0),
    )[mask]
    return (pair_losses * pair_weights).sum() / pair_weights.sum().clamp_min(1e-9)


def spread_regression_loss(pred_spread, true_spread, beta, sample_weights=None):
    """Robustly match each seed set's normalized total spread."""
    losses = F.smooth_l1_loss(
        pred_spread,
        true_spread,
        beta=beta,
        reduction='none',
    )
    if sample_weights is None:
        return losses.mean()
    return (losses * sample_weights).sum() / sample_weights.sum().clamp_min(1e-9)


def tail_sample_weights(true_spread_fraction, args):
    """Assign fixed-data upper-tail weights without generating new labels."""
    tail_threshold = float(args.tail_spread_threshold)
    extreme_threshold = float(args.extreme_spread_threshold)
    weights = torch.ones_like(true_spread_fraction)
    weights = torch.where(
        true_spread_fraction >= tail_threshold,
        weights.new_full((), float(args.tail_weight)),
        weights,
    )
    return torch.where(
        true_spread_fraction >= extreme_threshold,
        weights.new_full((), float(args.extreme_weight)),
        weights,
    )


def _safe_correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if (
            left.size < 2
            or np.isclose(left.std(), 0.0)
            or np.isclose(right.std(), 0.0)):
        return None
    correlation = float(np.corrcoef(left, right)[0, 1])
    return correlation if np.isfinite(correlation) else None


def _spread_diagnostics(predicted_spreads, true_spreads):
    predicted_spreads = np.asarray(predicted_spreads, dtype=np.float64)
    true_spreads = np.asarray(true_spreads, dtype=np.float64)
    return {
        'spread_mae': float(np.abs(
            predicted_spreads - true_spreads
        ).mean()),
        'spread_pearson': _safe_correlation(
            predicted_spreads, true_spreads
        ),
        'spread_spearman': _safe_correlation(
            rankdata(predicted_spreads),
            rankdata(true_spreads),
        ),
    }


def _format_correlation(value):
    return 'n/a' if value is None else f'{value:.4f}'


def checkpoint_is_better(
        candidate_spearman,
        candidate_loss,
        best_spearman,
        best_loss,
):
    """Prefer validation ranking quality, using loss only as a tie-breaker."""
    candidate_rank = (
        -float('inf')
        if candidate_spearman is None or not np.isfinite(candidate_spearman)
        else float(candidate_spearman)
    )
    best_rank = (
        -float('inf')
        if best_spearman is None or not np.isfinite(best_spearman)
        else float(best_spearman)
    )
    return (
        candidate_rank > best_rank
        or (candidate_rank == best_rank and candidate_loss < best_loss)
    )


def train_model(
        forward_model,
        optimizer,
        train_loader,
        val_loader,
        adjacency,
        num_nodes,
        device,
        args,
        checkpoint_file,
):
    print("\n[1/4] 训练")

    best_val_spearman = None
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    
    for epoch in range(args.epochs):
        forward_model.train()
        train_mse_sum = 0
        train_spread_sum = 0
        train_rank_sum = 0
        optimizer.zero_grad(set_to_none=True)
    
        for batch_idx, data_pair in enumerate(train_loader):
            x, y = unpack_batch(data_pair, num_nodes, device)
            B = x.size(0)
    
            with autocast_context(args.amp_dtype, device):
                y_hat_batch = forward_sample_batch(
                    forward_model,
                    x,
                    adjacency,
                    args.model_micro_batch_size
                )
                # Ranking loss uses a normalized score only for scale stability;
                # the reported Total Spread metric remains an absolute node count.
                pred_spread_fraction = y_hat_batch.sum(dim=1) / num_nodes
                true_spread_fraction = y.sum(dim=1) / num_nodes
                sample_weights = tail_sample_weights(
                    true_spread_fraction, args
                )
                per_sample_mse = F.mse_loss(
                    y_hat_batch, y, reduction='none'
                ).mean(dim=1)
                mse_loss = (
                    per_sample_mse * sample_weights
                ).sum() / sample_weights.sum().clamp_min(1e-9)
                spread_loss = spread_regression_loss(
                    pred_spread_fraction,
                    true_spread_fraction,
                    args.spread_beta,
                    sample_weights=sample_weights,
                )
                rank_loss = pairwise_ranking_loss(
                    pred_spread_fraction,
                    true_spread_fraction,
                    args.ranking_margin,
                    args.ranking_temperature,
                    sample_weights=sample_weights,
                )
                loss = (
                    mse_loss
                    + args.spread_weight * spread_loss
                    + args.ranking_weight * rank_loss
                )
    
            (loss / args.grad_accum_steps).backward()
            should_step = (
                (batch_idx + 1) % args.grad_accum_steps == 0
                or batch_idx + 1 == len(train_loader)
            )
            if should_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    
            train_mse_sum += mse_loss.item()
            train_spread_sum += spread_loss.item()
            train_rank_sum += rank_loss.item()
    
        forward_model.eval()
        val_mse_sum = 0
        val_spread_sum = 0
        val_rank_sum = 0
        val_predicted_spreads = []
        val_true_spreads = []
    
        with torch.no_grad():
            for data_pair in val_loader:
                x, y = unpack_batch(data_pair, num_nodes, device)
                B = x.size(0)
    
                with autocast_context(args.amp_dtype, device):
                    y_hat_batch = forward_sample_batch(
                        forward_model,
                        x,
                        adjacency,
                        args.model_micro_batch_size
                    )
                    val_mse_sum += F.mse_loss(y_hat_batch, y).item()
                    pred_spread_fraction = y_hat_batch.sum(dim=1) / num_nodes
                    true_spread_fraction = y.sum(dim=1) / num_nodes
                    val_spread_sum += spread_regression_loss(
                        pred_spread_fraction,
                        true_spread_fraction,
                        args.spread_beta,
                    ).item()
                    val_rank_sum += pairwise_ranking_loss(
                        pred_spread_fraction,
                        true_spread_fraction,
                        args.ranking_margin,
                        args.ranking_temperature,
                    ).item()
                    val_predicted_spreads.extend(
                        (pred_spread_fraction * num_nodes).float().cpu().tolist()
                    )
                    val_true_spreads.extend(
                        (true_spread_fraction * num_nodes).float().cpu().tolist()
                    )
    
        avg_train_mse = train_mse_sum / len(train_loader)
        avg_train_spread = train_spread_sum / len(train_loader)
        avg_train_rank = train_rank_sum / len(train_loader)
        avg_val_mse = val_mse_sum / len(val_loader)
        avg_val_spread = val_spread_sum / len(val_loader)
        avg_val_rank = val_rank_sum / len(val_loader)
        avg_val_loss = (
            avg_val_mse
            + args.spread_weight * avg_val_spread
            + args.ranking_weight * avg_val_rank
        )
        val_diagnostics = _spread_diagnostics(
            val_predicted_spreads,
            val_true_spreads,
        )
    
        epoch_number = epoch + 1
        final_epoch = args.epochs
        current_val_spearman = val_diagnostics['spread_spearman']
        scheduler_metric = (
            current_val_spearman
            if current_val_spearman is not None
            else -avg_val_loss
        )
        scheduler.step(scheduler_metric)
        current_lr = float(optimizer.param_groups[0]['lr'])

        if epoch_number == 1 or epoch_number % 10 == 0:
            print(
                f"轮次 {epoch_number}/{final_epoch} | "
                f"训练MSE {avg_train_mse:.6f} | "
                f"验证MSE {avg_val_mse:.6f} | "
                "Spearman "
                f"{_format_correlation(current_val_spearman)} | "
                f"LR {current_lr:.2e}"
            )
    
        if checkpoint_is_better(
                current_val_spearman,
                avg_val_loss,
                best_val_spearman,
                best_val_loss):
            best_val_loss = avg_val_loss
            best_val_spearman = current_val_spearman
            best_epoch = epoch_number
            patience_counter = 0
            torch.save({
                'format_version': 5,
                'torch_version': str(torch.__version__),
                'model_type': 'MCATjl',
                'model_config': {
                    'nfeat': 1,
                    'nhid': 64,
                    'nclass': 1,
                    'dropout': 0.2,
                    'nheads': 4,
                    'alpha': 0.2,
                    'structural_features': [
                        'degree',
                        'clustering_coefficient',
                    ],
                    'positive_mapping': 'softplus',
                    'attention_edge_bias': True,
                    'output_bias_init': float(args.output_bias_init),
                    'seed_probability_constraint': True,
                    'global_seed_pooling': {
                        'features': [
                            'raw_degree',
                            'log_degree',
                            'pagerank',
                            'clustering_coefficient',
                        ],
                        'hidden_size': int(args.global_pool_hidden),
                        'aggregation': 'seed_weighted_mean',
                        'target': 'node_logits',
                    },
                },
                'objective_config': {
                    'node_mse_weight': 1.0,
                    'spread_weight': float(args.spread_weight),
                    'spread_beta': float(args.spread_beta),
                    'ranking_weight': float(args.ranking_weight),
                    'ranking_tie_threshold': float(args.ranking_margin),
                    'ranking_temperature': float(args.ranking_temperature),
                    'upper_tail_weights': {
                        'tail_quantile': float(args.tail_quantile),
                        'extreme_quantile': float(args.extreme_quantile),
                        'tail_spread_threshold': float(
                            args.tail_spread_threshold
                        ),
                        'extreme_spread_threshold': float(
                            args.extreme_spread_threshold
                        ),
                        'tail_weight': float(args.tail_weight),
                        'extreme_weight': float(args.extreme_weight),
                    },
                },
                'epoch': epoch_number,
                'checkpoint_selection': {
                    'primary': 'val_spread_spearman',
                    'tie_breaker': 'val_combined_loss',
                },
                'forward_state_dict': forward_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'learning_rate': current_lr,
                'val_mse': avg_val_mse,
                'val_spread_spearman': current_val_spearman,
                'val_spread_loss': avg_val_spread,
                'val_rank': avg_val_rank,
                'val_spread_diagnostics': val_diagnostics,
            }, checkpoint_file)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"早停: {epoch_number} | 最佳: {best_epoch}")
                break
    
    print(
        f"训练完成 | 最佳轮次: {best_epoch} | "
        f"Spearman: {_format_correlation(best_val_spearman)} | "
        f"损失: {best_val_loss:.6f}"
    )
    return best_epoch, best_val_loss, best_val_spearman


def evaluate_predictor(
        forward_model,
        test_loader,
        test_set_size,
        adjacency,
        num_nodes,
        device,
        args,
        checkpoint_file,
):
    print("\n[2/4] 测试")
    
    checkpoint = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=True,
    )
    forward_model.load_state_dict(checkpoint['forward_state_dict'])
    forward_model.eval()
    
    test_mse_sum = 0.0
    target_square_sum = 0.0
    target_value_count = 0
    prediction_min = float('inf')
    prediction_max = -float('inf')
    prediction_sum = 0.0
    prediction_value_count = 0
    predicted_spreads = []
    true_spreads = []
    with torch.no_grad():
        for data_pair in test_loader:
            x, y = unpack_batch(data_pair, num_nodes, device)
            with autocast_context(args.amp_dtype, device):
                y_hat_batch = forward_sample_batch(
                    forward_model,
                    x,
                    adjacency,
                    args.model_micro_batch_size
                )
                per_sample_mse = F.mse_loss(
                    y_hat_batch, y, reduction='none'
                ).mean(dim=1)
                test_mse_sum += per_sample_mse.sum().item()
                target_square_sum += y.square().sum().item()
                target_value_count += y.numel()
                prediction_min = min(
                    prediction_min, float(y_hat_batch.min().item())
                )
                prediction_max = max(
                    prediction_max, float(y_hat_batch.max().item())
                )
                prediction_sum += y_hat_batch.sum().item()
                prediction_value_count += y_hat_batch.numel()
                predicted_spreads.extend(
                    y_hat_batch.sum(dim=1).float().cpu().tolist()
                )
                true_spreads.extend(y.sum(dim=1).float().cpu().tolist())

    predicted_spreads = np.asarray(predicted_spreads, dtype=np.float64)
    true_spreads = np.asarray(true_spreads, dtype=np.float64)

    spread_diagnostics = _spread_diagnostics(
        predicted_spreads,
        true_spreads,
    )
    pearson = spread_diagnostics['spread_pearson']
    spearman = spread_diagnostics['spread_spearman']
    metrics = {
        'mse': float(test_mse_sum / test_set_size),
        'zero_predictor_mse': float(
            target_square_sum / max(1, target_value_count)
        ),
        'prediction_min': float(prediction_min),
        'prediction_mean': float(
            prediction_sum / max(1, prediction_value_count)
        ),
        'prediction_max': float(prediction_max),
        'predicted_spread_mean': float(predicted_spreads.mean()),
        'true_spread_mean': float(true_spreads.mean()),
        'spread_mae': spread_diagnostics['spread_mae'],
        'spread_pearson': pearson,
        'spread_spearman': spearman,
    }

    print(
        f"MSE: {metrics['mse']:.6f} | "
        f"Spread MAE: {metrics['spread_mae']:.2f} | "
        f"Pearson: {_format_correlation(pearson)} | "
        f"Spearman: {_format_correlation(spearman)}"
    )
    return metrics
