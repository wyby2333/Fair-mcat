"""Command-line parsing and validation for Fair-mcat."""

import argparse
import random

import numpy as np
import torch

from data.config import DIFFUSION_FILE_SUFFIX, load_dataset_config


def parse_experiment_args(device):
    parser = argparse.ArgumentParser(description="Fair-mcat")
    parser.add_argument("-d", "--dataset", default="email", type=str)
    parser.add_argument("--config", type=str, help="自定义数据集配置 JSON")
    parser.add_argument(
        "--data_file", type=str,
        help=f"覆盖默认 {DIFFUSION_FILE_SUFFIX} 路径",
    )
    parser.add_argument("--checkpoint", type=str, help="模型检查点路径")
    parser.add_argument("--result_file", type=str, help="结果 JSON 路径")
    parser.add_argument(
        "-dm", "--diffusion_model",
        choices=["IC", "LT"], default=None,
    )
    parser.add_argument("--ic_prob", default=None, type=float)
    parser.add_argument(
        "-sn", "--seed_num", default=10, type=int,
        choices=[5, 10, 20, 30, 40, 50],
    )

    parser.add_argument("--random_seed", default=None, type=int)
    parser.add_argument("--train_ratio", default=None, type=float)
    parser.add_argument("--val_ratio", default=None, type=float)
    parser.add_argument("--test_ratio", default=None, type=float)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--patience", default=None, type=int)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--lr_patience", default=None, type=int)
    parser.add_argument("--lr_factor", default=None, type=float)
    parser.add_argument("--min_lr", default=None, type=float)
    parser.add_argument("--output_bias_init", default=None, type=float)
    parser.add_argument("--global_pool_hidden", default=None, type=int)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--grad_accum_steps", default=None, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument(
        "--amp_dtype", choices=["off", "bfloat16"], default=None,
    )
    parser.add_argument("--model_micro_batch_size", default=None, type=int)
    parser.add_argument("--edge_chunk_size", default=None, type=int)
    parser.add_argument(
        "--graph_batch_backend",
        choices=["block", "shared", "auto"],
        default=None,
    )
    parser.add_argument("--ranking_weight", default=None, type=float)
    parser.add_argument("--ranking_margin", default=None, type=float)
    parser.add_argument("--ranking_temperature", default=None, type=float)
    parser.add_argument("--spread_weight", default=None, type=float)
    parser.add_argument("--spread_beta", default=None, type=float)
    parser.add_argument("--tail_quantile", default=None, type=float)
    parser.add_argument("--extreme_quantile", default=None, type=float)
    parser.add_argument("--tail_weight", default=None, type=float)
    parser.add_argument("--extreme_weight", default=None, type=float)

    parser.add_argument("--num_simulations", default=None, type=int)
    parser.add_argument("--evaluation_seed", default=314159, type=int)
    parser.add_argument(
        "--candidate_mc_initial_runs", default=None, type=int,
    )
    parser.add_argument(
        "--candidate_mc_confirm_runs", default=None, type=int,
    )
    parser.add_argument(
        "--candidate_mc_finalist_count", default=None, type=int,
    )

    parser.add_argument("--search_steps", default=None, type=int)
    parser.add_argument("--num_restarts", default=None, type=int)
    parser.add_argument("--patience_steps", default=None, type=int)
    parser.add_argument(
        "--search_execution",
        choices=["sequential", "batched"],
        default=None,
    )
    parser.add_argument(
        "--search_spread_floor_ratio", default=None, type=float,
    )
    parser.add_argument("--fairness_strength", default=0.7, type=float)

    args = parser.parse_args()
    canonical_dataset, dataset_config = load_dataset_config(
        args.dataset, args.config,
    )
    args.dataset = canonical_dataset

    config_defaults = {
        "diffusion_model": "diffusion_model",
        "ic_prob": "ic_prob",
        "random_seed": "random_seed",
        "train_ratio": "train_ratio",
        "val_ratio": "val_ratio",
        "test_ratio": "test_ratio",
        "epochs": "epochs",
        "patience": "patience",
        "lr": "lr",
        "lr_patience": "lr_patience",
        "lr_factor": "lr_factor",
        "min_lr": "min_lr",
        "output_bias_init": "output_bias_init",
        "global_pool_hidden": "global_pool_hidden",
        "batch_size": "batch_size",
        "grad_accum_steps": "grad_accum_steps",
        "amp_dtype": "amp_dtype",
        "model_micro_batch_size": "model_micro_batch_size",
        "edge_chunk_size": "edge_chunk_size",
        "graph_batch_backend": "graph_batch_backend",
        "ranking_weight": "ranking_weight",
        "ranking_margin": "ranking_margin",
        "ranking_temperature": "ranking_temperature",
        "spread_weight": "spread_weight",
        "spread_beta": "spread_beta",
        "tail_quantile": "tail_quantile",
        "extreme_quantile": "extreme_quantile",
        "tail_weight": "tail_weight",
        "extreme_weight": "extreme_weight",
        "num_simulations": "final_mc_runs",
        "candidate_mc_initial_runs": "candidate_mc_initial_runs",
        "candidate_mc_confirm_runs": "candidate_mc_confirm_runs",
        "candidate_mc_finalist_count": "candidate_mc_finalist_count",
        "search_steps": "search_steps",
        "num_restarts": "num_restarts",
        "patience_steps": "patience_steps",
        "search_execution": "search_execution",
        "search_spread_floor_ratio": "search_spread_floor_ratio",
    }
    fallbacks = {
        "graph_batch_backend": "block",
        "global_pool_hidden": 16,
        "lr_patience": 20,
        "lr_factor": 0.5,
        "min_lr": 1e-5,
        "tail_quantile": 0.8,
        "extreme_quantile": 0.95,
        "tail_weight": 2.0,
        "extreme_weight": 3.0,
        "candidate_mc_initial_runs": 20,
        "candidate_mc_confirm_runs": 100,
        "candidate_mc_finalist_count": 2,
        "search_spread_floor_ratio": 0.98,
    }
    for argument_name, config_name in config_defaults.items():
        if getattr(args, argument_name) is not None:
            continue
        value = dataset_config.get(config_name, fallbacks.get(config_name))
        if value is None:
            raise KeyError(f"配置缺少字段: {config_name}")
        setattr(args, argument_name, value)

    if not np.isclose(
        args.train_ratio + args.val_ratio + args.test_ratio, 1.0,
    ):
        raise ValueError("train_ratio、val_ratio、test_ratio 的总和必须为1。")
    if args.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps 必须大于或等于1。")
    if args.num_workers < 0:
        raise ValueError("num_workers 不能小于0。")
    if args.model_micro_batch_size < 0:
        raise ValueError("model_micro_batch_size 不能小于0。")
    if args.edge_chunk_size < 0:
        raise ValueError("edge_chunk_size 不能小于0。")
    if args.global_pool_hidden < 1:
        raise ValueError("global_pool_hidden 必须大于或等于1。")
    if args.spread_weight < 0 or args.ranking_weight < 0:
        raise ValueError("损失权重不能小于0。")
    if args.lr <= 0 or args.min_lr < 0 or args.min_lr > args.lr:
        raise ValueError("学习率必须满足 0 <= min_lr <= lr 且 lr > 0。")
    if args.lr_patience < 1:
        raise ValueError("lr_patience 必须大于或等于1。")
    if not 0.0 < args.lr_factor < 1.0:
        raise ValueError("lr_factor 必须位于0和1之间。")
    if args.spread_beta <= 0:
        raise ValueError("spread_beta 必须大于0。")
    if args.ranking_margin < 0:
        raise ValueError("ranking_margin 不能小于0。")
    if args.ranking_temperature <= 0:
        raise ValueError("ranking_temperature 必须大于0。")
    if not 0.0 < args.tail_quantile < args.extreme_quantile < 1.0:
        raise ValueError(
            "必须满足 0 < tail_quantile < extreme_quantile < 1。"
        )
    if args.tail_weight < 1.0 or args.extreme_weight < args.tail_weight:
        raise ValueError(
            "尾部权重必须满足 1 <= tail_weight <= extreme_weight。"
        )
    if not 0.0 < args.search_spread_floor_ratio <= 1.0:
        raise ValueError("search_spread_floor_ratio 必须位于0和1之间。")
    if not (
        1 <= args.candidate_mc_initial_runs
        <= args.candidate_mc_confirm_runs
    ):
        raise ValueError(
            "候选评估次数必须满足 1 <= initial_runs <= confirm_runs。"
        )
    if args.candidate_mc_finalist_count < 1:
        raise ValueError(
            "candidate_mc_finalist_count 必须大于或等于1。"
        )
    if args.num_simulations < 1:
        raise ValueError("num_simulations 必须大于或等于1。")
    if not 0.0 <= args.ic_prob <= 1.0:
        raise ValueError("ic_prob 必须位于0~1。")
    if not 0.0 <= args.fairness_strength <= 1.0:
        raise ValueError("fairness_strength 必须位于0~1。")

    if args.amp_dtype == "bfloat16":
        if device.type != "cuda":
            print("[提示] 使用 FP32。")
            args.amp_dtype = "off"
        elif not torch.cuda.is_bf16_supported():
            print("[提示] 使用 FP32。")
            args.amp_dtype = "off"

    random_seed = int(args.random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    return args, dataset_config, random_seed

