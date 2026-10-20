import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


POSITIVE_MAPPING = "softplus"


def _positive_weight(parameter):
    """Map a trainable raw parameter to a smooth nonnegative weight."""
    return F.softplus(parameter)


def _initialize_positive_raw_(parameter):
    """Preserve Xavier/Linear magnitudes after applying softplus."""
    with torch.no_grad():
        effective = parameter.abs().clamp_min(1e-4)
        raw = effective + torch.log(-torch.expm1(-effective))
        parameter.copy_(raw)


class SpecialSpmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        values_dtype = values.dtype
        b_dtype = b.dtype
        compute_dtype = torch.promote_types(values_dtype, b_dtype)
        # CUDA sparse matmul does not implement all low-precision kernels
        # (notably BF16 on current Blackwell/PyTorch builds). Keep AMP for the
        # dense attention projections, but accumulate sparse messages in FP32.
        if compute_dtype in (torch.float16, torch.bfloat16):
            compute_dtype = torch.float32
        values_compute = values.to(dtype=compute_dtype)
        b_compute = b.to(dtype=compute_dtype)
        with torch.autocast(device_type=b.device.type, enabled=False):
            a = torch.sparse_coo_tensor(
                indices,
                values_compute,
                shape,
                check_invariants=False,
            )
            output = torch.matmul(a, b_compute)
        ctx.save_for_backward(a, b_compute)
        ctx.values_dtype = values_dtype
        ctx.b_dtype = b_dtype
        ctx.N = shape[0]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        grad_output = grad_output.to(dtype=b.dtype)
        with torch.autocast(device_type=b.device.type, enabled=False):
            if ctx.needs_input_grad[1]:
                row, col = a._indices()
                grad_values = (
                    grad_output.index_select(0, row)
                    * b.index_select(0, col)
                ).sum(dim=1).to(dtype=ctx.values_dtype)
            if ctx.needs_input_grad[3]:
                grad_b = a.t().matmul(grad_output).to(dtype=ctx.b_dtype)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SharedEdgeSpmmFunction(torch.autograd.Function):
    """Batched sparse aggregation with one shared edge-index tensor."""

    @staticmethod
    def forward(ctx, indices, values, num_nodes, features, chunk_size):
        if values.dim() != 2 or features.dim() != 3:
            raise ValueError("共享边聚合要求 values=[B,E]、features=[B,N,H]。")
        if values.size(0) != features.size(0):
            raise ValueError("共享边聚合的 batch 大小不一致。")

        values_dtype = values.dtype
        features_dtype = features.dtype
        compute_dtype = torch.promote_types(values_dtype, features_dtype)
        if compute_dtype in (torch.float16, torch.bfloat16):
            compute_dtype = torch.float32
        values_compute = values.to(dtype=compute_dtype)
        features_compute = features.to(dtype=compute_dtype)

        with torch.autocast(device_type=features.device.type, enabled=False):
            row, col = indices
            edge_count = int(indices.size(1))
            chunk_size = int(chunk_size or edge_count or 1)
            output = features_compute.new_zeros(
                (
                    features_compute.size(0),
                    int(num_nodes),
                    features_compute.size(2),
                )
            )
            for start in range(0, edge_count, chunk_size):
                stop = min(start + chunk_size, edge_count)
                rows = row[start:stop]
                cols = col[start:stop]
                messages = (
                    values_compute[:, start:stop].unsqueeze(-1)
                    * features_compute.index_select(1, cols)
                )
                output.index_add_(1, rows, messages)

        ctx.save_for_backward(indices, values_compute, features_compute)
        ctx.values_dtype = values_dtype
        ctx.features_dtype = features_dtype
        ctx.chunk_size = chunk_size
        return output

    @staticmethod
    def backward(ctx, grad_output):
        indices, values, features = ctx.saved_tensors
        row, col = indices
        edge_count = int(indices.size(1))
        grad_values = torch.empty_like(values) if ctx.needs_input_grad[1] else None
        grad_features = (
            torch.zeros_like(features) if ctx.needs_input_grad[3] else None
        )

        grad_output = grad_output.to(dtype=features.dtype)
        with torch.autocast(device_type=features.device.type, enabled=False):
            for start in range(0, edge_count, ctx.chunk_size):
                stop = min(start + ctx.chunk_size, edge_count)
                rows = row[start:stop]
                cols = col[start:stop]
                selected_grad = grad_output.index_select(1, rows)
                if grad_values is not None:
                    grad_values[:, start:stop] = (
                        selected_grad * features.index_select(1, cols)
                    ).sum(dim=-1)
                if grad_features is not None:
                    grad_messages = (
                        values[:, start:stop].unsqueeze(-1) * selected_grad
                    )
                    grad_features.index_add_(1, cols, grad_messages)

        if grad_values is not None:
            grad_values = grad_values.to(dtype=ctx.values_dtype)
        if grad_features is not None:
            grad_features = grad_features.to(dtype=ctx.features_dtype)

        return None, grad_values, None, grad_features, None


class SharedEdgeSpmm(nn.Module):
    def forward(self, indices, values, num_nodes, features, chunk_size=0):
        return SharedEdgeSpmmFunction.apply(
            indices,
            values,
            int(num_nodes),
            features,
            int(chunk_size or 0),
        )


class SpGraphTransformerLayer(nn.Module):
    """Graph Transformer layer with monotonic-consistency enhancement.

    Smooth softplus mappings applied to W_V and W_O restrict negative feature
    contributions without the dead-gradient region of ReLU. They encourage
    consistency with diffusion monotonicity but do not constitute a proof that
    the full attention network is monotonic.
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True,
                 edge_chunk_size=0):
        super(SpGraphTransformerLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.edge_chunk_size = int(edge_chunk_size or 0)
        self.dropout = nn.Dropout(dropout)

        self.W_Q = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.W_K = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.W_V = nn.Parameter(torch.empty(size=(in_features, out_features)))

        nn.init.xavier_normal_(self.W_Q, gain=1.414)
        nn.init.xavier_normal_(self.W_K, gain=1.414)
        nn.init.xavier_normal_(self.W_V, gain=1.414)

        self.W_O = nn.Linear(out_features, out_features, bias=False)

        _initialize_positive_raw_(self.W_V)
        _initialize_positive_raw_(self.W_O.weight)
        self.edge_bias = nn.Parameter(torch.zeros(1))

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
        self.shared_edge_spmm = SharedEdgeSpmm()

    def forward(self, input, adj=None, edge_index=None, num_nodes=None):
        if input.dim() == 3:
            return self._forward_shared(
                input,
                adj=adj,
                edge_index=edge_index,
                num_nodes=num_nodes,
            )
        return self._forward_block(
            input,
            adj=adj,
            edge_index=edge_index,
            num_nodes=num_nodes,
        )

    def _forward_block(self, input, adj=None, edge_index=None, num_nodes=None):
        N = int(num_nodes or input.size(0))

        if edge_index is not None:
            edge = edge_index
        elif adj is not None and adj.layout == torch.sparse_coo:
            edge = adj.indices()
        elif adj is not None:
            edge = adj.nonzero().t()
        else:
            raise ValueError("adj 和 edge_index 至少需要提供一个。")

        Q = torch.mm(input, self.W_Q)
        K = torch.mm(input, self.W_K)
        V = torch.mm(input, _positive_weight(self.W_V))

        scale_factor = self.out_features ** 0.5
        edge_count = edge.size(1)
        if self.edge_chunk_size > 0 and edge_count > self.edge_chunk_size:
            dot_product_chunks = []
            for start in range(0, edge_count, self.edge_chunk_size):
                stop = min(start + self.edge_chunk_size, edge_count)
                rows = edge[0, start:stop]
                cols = edge[1, start:stop]
                dot_product_chunks.append(
                    (
                        Q.index_select(0, rows)
                        * K.index_select(0, cols)
                    ).sum(dim=1) / scale_factor
                )
            dot_prod = torch.cat(dot_product_chunks, dim=0)
        else:
            Q_edges = Q.index_select(0, edge[0])
            K_edges = K.index_select(0, edge[1])
            dot_prod = (Q_edges * K_edges).sum(dim=1) / scale_factor

        edge_e = torch.exp(torch.clamp(
            self.leakyrelu(dot_prod + self.edge_bias), max=50
        ))

        e_rowsum = self.special_spmm(
            edge,
            edge_e,
            torch.Size([N, N]),
            torch.ones(size=(N, 1), dtype=edge_e.dtype, device=input.device)
        )

        edge_e = self.dropout(edge_e)

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), V)
        h_prime = h_prime.div(e_rowsum + 1e-9)

        if self.concat:
            h = F.elu(h_prime)
            return F.linear(
                h, _positive_weight(self.W_O.weight), self.W_O.bias
            )
        else:
            return h_prime

    def _forward_shared(self, input, adj=None, edge_index=None, num_nodes=None):
        batch_size, input_nodes, _ = input.shape
        N = int(num_nodes or input_nodes)
        if N != input_nodes:
            raise ValueError("共享边模式的 num_nodes 必须等于每个样本的节点数。")

        if edge_index is not None:
            edge = edge_index
        elif adj is not None and adj.layout == torch.sparse_coo:
            edge = adj.coalesce().indices()
        elif adj is not None:
            edge = adj.nonzero().t().contiguous()
        else:
            raise ValueError("adj 和 edge_index 至少需要提供一个。")

        Q = torch.matmul(input, self.W_Q)
        K = torch.matmul(input, self.W_K)
        V = torch.matmul(input, _positive_weight(self.W_V))

        scale_factor = self.out_features ** 0.5
        edge_count = int(edge.size(1))
        chunk_size = int(self.edge_chunk_size or edge_count or 1)
        dot_product_chunks = []
        for start in range(0, edge_count, chunk_size):
            stop = min(start + chunk_size, edge_count)
            rows = edge[0, start:stop]
            cols = edge[1, start:stop]
            dot_product_chunks.append(
                (
                    Q.index_select(1, rows)
                    * K.index_select(1, cols)
                ).sum(dim=-1) / scale_factor
            )
        dot_prod = torch.cat(dot_product_chunks, dim=1)

        edge_e = torch.exp(
            torch.clamp(self.leakyrelu(dot_prod + self.edge_bias), max=50)
        )

        ones = torch.ones(
            (batch_size, N, 1),
            dtype=edge_e.dtype,
            device=input.device,
        )
        e_rowsum = self.shared_edge_spmm(
            edge, edge_e, N, ones, chunk_size
        )

        # Flatten in batch-major order so dropout consumes random values in the
        # same logical order as the block-diagonal implementation.
        dropped_edge_e = self.dropout(edge_e.reshape(-1)).view_as(edge_e)
        h_prime = self.shared_edge_spmm(
            edge, dropped_edge_e, N, V, chunk_size
        )
        h_prime = h_prime.div(e_rowsum + 1e-9)

        if self.concat:
            h = F.elu(h_prime)
            return F.linear(
                h, _positive_weight(self.W_O.weight), self.W_O.bias
            )
        return h_prime


class MCATjl(nn.Module):
    """
    MCAT + 局部聚类编码 + 种子集合级全局结构偏移。

    全局分支对种子指示量加权池化原始度、对数度、PageRank 和聚类
    系数，并将小型 MLP 的输出广播加到每个节点的激活 logit。
    """

    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads,
                 raw_adj=None, edge_chunk_size=0, degree_values=None,
                 clustering_coefficients=None, graph_batch_backend="block",
                 output_bias_init=-3.0, pagerank_scores=None,
                 global_pool_hidden=16):
        super(MCATjl, self).__init__()
        self.dropout = dropout
        self.edge_chunk_size = int(edge_chunk_size or 0)
        if graph_batch_backend not in {"block", "shared", "auto"}:
            raise ValueError("graph_batch_backend 必须是 block、shared 或 auto。")
        self.graph_batch_backend = graph_batch_backend
        self.supports_batched_graph = True
        self.output_bias_init = float(output_bias_init)
        self.output_bias = nn.Parameter(torch.full(
            (nclass,), self.output_bias_init, dtype=torch.float32
        ))

        # The graph is fixed for the full train/search run. The non-persistent
        # edge cache avoids repeated extraction while the raw-graph degree
        # feature remains separate from the normalized propagation adjacency.
        self.register_buffer(
            '_cached_edge_index',
            torch.empty((2, 0), dtype=torch.long),
            persistent=False
        )
        self._cached_num_nodes = 0
        self._batched_edge_cache = {}
        self.register_buffer(
            '_cached_degree_encoding',
            torch.empty((0, nfeat), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            '_cached_clustering_encoding',
            torch.empty((0, nfeat), dtype=torch.float32),
            persistent=False,
        )

        if degree_values is None and raw_adj is None:
            raise ValueError("必须提供 degree_values 或 raw_adj。")
        if clustering_coefficients is None or pagerank_scores is None:
            raise ValueError("必须提供 clustering_coefficients 和 pagerank_scores。")
        raw_degree = (
            np.asarray(degree_values).reshape(-1)
            if degree_values is not None
            else np.asarray(raw_adj.sum(axis=1)).reshape(-1)
        )
        clustering = np.asarray(
            clustering_coefficients, dtype=np.float32
        ).reshape(-1)
        pagerank = np.asarray(pagerank_scores, dtype=np.float32).reshape(-1)
        if not (raw_degree.size == pagerank.size == clustering.size):
            raise ValueError("图特征的节点数不一致。")

        self.register_buffer(
            '_degree_log',
            torch.tensor(
                np.log1p(raw_degree), dtype=torch.float32
            ).view(-1, 1),
            persistent=False
        )
        self.centrality_encoder = nn.Sequential(
            nn.Linear(1, nfeat), nn.ReLU(), nn.Linear(nfeat, nfeat)
        )
        self.register_buffer(
            'clustering_coef',
            torch.as_tensor(clustering, dtype=torch.float32).reshape(-1, 1),
        )
        self.clustering_encoder = nn.Sequential(
            nn.Linear(1, nfeat), nn.ReLU(), nn.Linear(nfeat, nfeat)
        )

        global_features = np.stack(
            [
                raw_degree.astype(np.float32, copy=False),
                np.log1p(raw_degree).astype(np.float32, copy=False),
                pagerank,
                clustering,
            ],
            axis=1,
        )
        feature_mean = global_features.mean(axis=0, keepdims=True)
        feature_std = global_features.std(axis=0, keepdims=True)
        feature_std = np.where(feature_std > 1e-6, feature_std, 1.0)
        global_features = (global_features - feature_mean) / feature_std
        self.register_buffer(
            '_global_node_features',
            torch.as_tensor(global_features, dtype=torch.float32),
            persistent=False,
        )
        hidden = int(global_pool_hidden)
        if hidden < 1:
            raise ValueError("global_pool_hidden 必须大于或等于1。")
        self.global_spread_mlp = nn.Sequential(
            nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, nclass)
        )

        self.attentions = nn.ModuleList([
            SpGraphTransformerLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True,
                                     edge_chunk_size=self.edge_chunk_size)
            for _ in range(nheads)
        ])
        self.out_att = SpGraphTransformerLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False,
                                                edge_chunk_size=self.edge_chunk_size)

    def cache_graph(self, adj):
        """Cache message-passing edges without redefining structural degree."""
        if adj.layout == torch.sparse_coo:
            adj = adj.coalesce()
            edge_index = adj.indices()
        else:
            edge_index = adj.nonzero().t().contiguous()

        self._cached_edge_index = edge_index.detach()
        self._cached_num_nodes = int(adj.shape[0])
        self._batched_edge_cache.clear()
        return self

    def _apply(self, fn):
        # Batched edge tensors are derived device-specific cache entries.
        self._batched_edge_cache.clear()
        self.clear_inference_cache()
        return super(MCATjl, self)._apply(fn)

    def train(self, mode=True):
        if mode:
            self.clear_inference_cache()
        return super(MCATjl, self).train(mode)

    def clear_inference_cache(self):
        for name in (
                '_cached_degree_encoding',
                '_cached_clustering_encoding'):
            cached = getattr(self, name)
            setattr(self, name, torch.empty(
                (0, cached.size(1)),
                dtype=cached.dtype,
                device=cached.device,
            ))

    def prepare_inference_cache(self):
        if self.training:
            raise RuntimeError("只能在 model.eval() 后准备静态推理缓存。")
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise RuntimeError("准备静态推理缓存前必须冻结模型参数。")
        with torch.no_grad():
            self._cached_degree_encoding = self.centrality_encoder(
                self._degree_log
            ).detach()
            self._cached_clustering_encoding = self.clustering_encoder(
                self.clustering_coef
            ).detach()
        return self

    def clear_graph_cache(self):
        self._cached_edge_index = torch.empty(
            (2, 0), dtype=torch.long, device=self._cached_edge_index.device
        )
        self._cached_num_nodes = 0
        self._batched_edge_cache.clear()

    def _resolve_graph(self, adj, x):
        if self._cached_edge_index.numel() > 0:
            if adj is not None and int(adj.shape[0]) != self._cached_num_nodes:
                raise ValueError("输入邻接矩阵与已缓存图的节点数不一致。")
            return self._cached_edge_index

        if adj is None:
            raise ValueError("尚未缓存图，forward 必须提供 adj。")
        if adj.layout == torch.sparse_coo:
            adj = adj.coalesce()
            edge_index = adj.indices()
        else:
            edge_index = adj.nonzero().t().contiguous()
        return edge_index

    def _batch_edge_index(self, edge_index, num_nodes, batch_size):
        if batch_size == 1:
            return edge_index

        cache_key = (
            int(batch_size),
            int(num_nodes),
            edge_index.device.type,
            edge_index.device.index
        )
        cached = self._batched_edge_cache.get(cache_key)
        if cached is not None:
            return cached

        offsets = (
            torch.arange(batch_size, device=edge_index.device)
            .view(batch_size, 1, 1)
            .mul(num_nodes)
        )
        batched = (
            edge_index.unsqueeze(0) + offsets
        ).permute(1, 0, 2).reshape(2, -1).contiguous()
        self._batched_edge_cache[cache_key] = batched
        return batched

    def _pool_global_seed_features(self, seed_indicator):
        """Differentiably pool structural features for every seed set."""
        seed_weights = seed_indicator.squeeze(-1)
        seed_mass = seed_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.matmul(
            seed_weights,
            self._global_node_features,
        ) / seed_mass

    def _global_seed_offset(self, seed_indicator):
        pooled_features = self._pool_global_seed_features(seed_indicator)
        return self.global_spread_mlp(pooled_features).unsqueeze(1)

    def forward(self, x, adj=None):
        squeeze_batch = x.dim() == 2
        if squeeze_batch:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError("MCATjl 输入必须是 [N,F] 或 [B,N,F]。")

        # A diffusion seed is active with probability one by definition. Keep
        # the original seed channel before adding static structural encodings,
        # then impose that known boundary condition on the final prediction.
        seed_indicator = x[..., :1].clamp(0.0, 1.0)

        batch_size, num_nodes, _ = x.shape
        edge_index = self._resolve_graph(adj, x)
        if self._degree_log.size(0) != num_nodes:
            raise ValueError("原始度特征与输入节点数不一致。")
        if self.clustering_coef.size(0) != num_nodes:
            raise ValueError("聚类系数特征与输入节点数不一致。")
        if self._global_node_features.size(0) != num_nodes:
            raise ValueError("全局种子特征与输入节点数不一致。")

        global_seed_offset = self._global_seed_offset(seed_indicator)

        degree_encoding = (
            self._cached_degree_encoding
            if self._cached_degree_encoding.numel() > 0
            else self.centrality_encoder(self._degree_log)
        )
        clustering_encoding = (
            self._cached_clustering_encoding
            if self._cached_clustering_encoding.numel() > 0
            else self.clustering_encoder(self.clustering_coef)
        )
        x = x + degree_encoding.unsqueeze(0)
        x = x + clustering_encoding.unsqueeze(0)

        x = F.dropout(x, self.dropout, training=self.training)
        backend = self.graph_batch_backend
        if backend == "auto":
            backend = "shared" if batch_size > 1 else "block"

        if backend == "shared":
            x = F.elu(torch.cat([
                att(x, edge_index=edge_index, num_nodes=num_nodes)
                for att in self.attentions
            ], dim=2))
            x = F.dropout(x, self.dropout, training=self.training)
            out = self.out_att(
                x,
                edge_index=edge_index,
                num_nodes=num_nodes,
            )
        else:
            batched_edge_index = self._batch_edge_index(
                edge_index, num_nodes, batch_size
            )
            x = x.reshape(batch_size * num_nodes, -1)
            x = F.elu(torch.cat([
                att(
                    x,
                    edge_index=batched_edge_index,
                    num_nodes=batch_size * num_nodes
                )
                for att in self.attentions
            ], dim=1))
            x = F.dropout(x, self.dropout, training=self.training)
            out = self.out_att(
                x,
                edge_index=batched_edge_index,
                num_nodes=batch_size * num_nodes
            ).view(batch_size, num_nodes, -1)
        # The supervision target is a node activation probability, so expose a
        # bounded probability rather than an unconstrained attention logit.
        logits = out + self.output_bias + global_seed_offset
        out = torch.sigmoid(logits)
        out = seed_indicator + (1.0 - seed_indicator) * out
        return out.squeeze(0) if squeeze_batch else out
