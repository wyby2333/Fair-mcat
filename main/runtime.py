"""Runtime helpers shared by Fair-MCAT training and search."""

from contextlib import nullcontext

import numpy as np
import scipy.sparse as sp
import torch


def normalize_adj(matrix):
    matrix = matrix.astype(np.float32).tocsr()
    row_sum = np.asarray(matrix.sum(1), dtype=np.float32)
    inverse_sqrt = np.power(row_sum, -0.5).flatten().astype(np.float32)
    inverse_sqrt[np.isinf(inverse_sqrt)] = 0.0
    inverse_sqrt_matrix = sp.diags(
        inverse_sqrt,
        dtype=np.float32,
    )
    return (
        matrix.dot(inverse_sqrt_matrix)
        .transpose()
        .dot(inverse_sqrt_matrix)
        .tocsr()
    )


def scipy_to_torch_sparse(matrix, target_device):
    """Convert scipy sparse data directly, without any dense N x N allocation."""
    coo = matrix.astype(np.float32).tocoo()
    indices = torch.from_numpy(
        np.vstack((coo.row, coo.col)).astype(np.int64, copy=False)
    )
    values = torch.from_numpy(coo.data.astype(np.float32, copy=False))
    tensor = torch.sparse_coo_tensor(
        indices,
        values,
        coo.shape,
        dtype=torch.float32,
        check_invariants=False,
    )
    return tensor.coalesce().to(target_device)


def unpack_batch(data_batch, num_nodes, target_device):
    """Create dense seed masks from an .IM v2 batch."""
    if not isinstance(data_batch, (list, tuple)) or len(data_batch) != 2:
        raise TypeError("数据批次必须包含 seed_indices 和 targets。")
    seed_indices, targets = data_batch
    seed_indices = seed_indices.long().to(target_device, non_blocking=True)
    targets = targets.float().to(target_device, non_blocking=True)
    batch_size = seed_indices.shape[0]
    seed_masks = torch.zeros(
        (batch_size, num_nodes),
        dtype=torch.float32,
        device=target_device,
    )
    seed_masks.scatter_(1, seed_indices, 1.0)
    return seed_masks, targets


def autocast_context(amp_dtype, device):
    if amp_dtype == "bfloat16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def forward_sample_batch(forward_model, seed_masks, adjacency, micro_batch_size=0):
    """Run a seed-set batch with adaptive CUDA micro-batching."""
    batch_size = int(seed_masks.size(0))
    requested_chunk_size = int(micro_batch_size or batch_size)
    cached_chunk_size = int(getattr(
        forward_model,
        "_effective_micro_batch_size",
        requested_chunk_size,
    ))
    chunk_size = min(batch_size, requested_chunk_size, cached_chunk_size)

    while True:
        outputs = []
        try:
            for start in range(0, batch_size, chunk_size):
                stop = min(start + chunk_size, batch_size)
                prediction = forward_model(
                    seed_masks[start:stop].unsqueeze(-1),
                    adjacency,
                ).squeeze(-1)
                outputs.append(prediction)
            return torch.cat(outputs, dim=0)
        except RuntimeError as error:
            is_cuda_oom = (
                seed_masks.is_cuda
                and "out of memory" in str(error).lower()
            )
            if not is_cuda_oom or chunk_size <= 1:
                raise
            outputs.clear()
            chunk_size = max(1, chunk_size // 2)
            forward_model._effective_micro_batch_size = chunk_size
            torch.cuda.empty_cache()
            print(f"[提示] micro-batch 调整为 {chunk_size}")
