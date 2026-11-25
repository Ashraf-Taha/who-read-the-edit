from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mutablekv.rope import phases, rotate, spin

ATTENTION_FLOAT_BUDGET = 48_000_000


@dataclass(frozen=True)
class Cells:
    """Cached state of a run of tokens: keys rotated at `enc_pos`, values, per layer."""

    tokens: torch.Tensor
    enc_pos: torch.Tensor
    K: torch.Tensor
    V: torch.Tensor

    def __len__(self) -> int:
        return self.tokens.numel()

    def take(self, index) -> Cells:
        on_device = index.to(self.K.device) if isinstance(index, torch.Tensor) else index
        return Cells(self.tokens[index], self.enc_pos[index], self.K[:, :, on_device], self.V[:, :, on_device])

    def replaced(self, chosen: torch.Tensor, fresh: Cells) -> Cells:
        """A copy in which the cells selected by the boolean mask `chosen` are swapped for `fresh`."""
        on_device = chosen.to(self.K.device)
        K, V, enc_pos = self.K.clone(), self.V.clone(), self.enc_pos.clone()
        K[:, :, on_device], V[:, :, on_device], enc_pos[chosen] = fresh.K, fresh.V, fresh.enc_pos
        return Cells(self.tokens, enc_pos, K, V)

    @staticmethod
    def cat(parts: list[Cells]) -> Cells:
        return Cells(
            torch.cat([p.tokens for p in parts]),
            torch.cat([p.enc_pos for p in parts]),
            torch.cat([p.K for p in parts], dim=2),
            torch.cat([p.V for p in parts], dim=2),
        )


@dataclass(frozen=True)
class Pass:
    """Output of `LM.run` for the fresh tokens.

    queries: per layer, the queries of the fresh tokens rotated at their own positions.
    lse: per layer and head, the log of each fresh token's softmax normaliser.
    Both are kept only when asked for; together they let one estimate later how much
    attention a token would have paid to content that did not exist when it was encoded.
    """

    cells: Cells
    hidden: torch.Tensor
    mass: torch.Tensor | None
    queries: torch.Tensor | None = None
    lse: torch.Tensor | None = None


class LM:
    """A causal LM of the Llama/Qwen2 family with the forward pass written out.

    `run` recomputes any subset of a sequence while reading cached cells for the
    rest, at arbitrary logical positions.
    """

    def __init__(self, repo: str, device: str = "cpu", dtype: torch.dtype = torch.float32):
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype).to(device).eval()
        self.repo, self.device, self.dtype = repo, device, dtype
        self.hf = model
        self.tok = AutoTokenizer.from_pretrained(repo)
        core, cfg = model.model, model.config
        self.embed, self.layers, self.final_norm, self.head = core.embed_tokens, core.layers, core.norm, model.lm_head
        self.L, self.H, self.G = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
        self.d_head = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        self.inv_freq = core.rotary_emb.inv_freq.detach().cpu().to(torch.float64)
        # In 16-bit runs, form attention scores and accumulate the weighted values in 32 bits,
        # as the library's fused attention does; rounding raw q.k products to bfloat16 first
        # loses most of their precision when they are large.
        self.wide_scores = True

    def rotate(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return rotate(x, pos, self.inv_freq)

    @staticmethod
    def _norm(module, x: torch.Tensor) -> torch.Tensor:
        # The library's RMSNorm casts to float32 internally, which injects float32
        # rounding into a float64 run and hides the exactness the tests check for.
        # In 16-bit runs we do cast, as the library does, so that results match it.
        if x.dtype in (torch.float16, torch.bfloat16):
            wide = x.float()
            return module.weight * (wide * torch.rsqrt(wide.pow(2).mean(-1, keepdim=True) + module.variance_epsilon)).to(x.dtype)
        return module.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + module.variance_epsilon))

    def unrotated(self, cells: Cells) -> torch.Tensor:
        return self.rotate(cells.K, -cells.enc_pos)

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(self.tok(text, add_special_tokens=False)["input_ids"], dtype=torch.long)

    @torch.inference_mode()
    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(self._norm(self.final_norm, hidden))

    def cold(self, tokens: torch.Tensor, pos: torch.Tensor | None = None, groups: torch.Tensor | None = None,
             keep_queries: bool = False) -> Pass:
        pos = torch.arange(tokens.numel()) if pos is None else pos
        return self.run(tokens, pos, groups=groups, keep_queries=keep_queries)

    @torch.inference_mode()
    def run(
        self,
        tokens: torch.Tensor,
        pos: torch.Tensor,
        fresh: torch.Tensor | None = None,
        cached: Cells | None = None,
        relocate: str = "query",
        groups: torch.Tensor | None = None,
        depth: int | None = None,
        keep_queries: bool = False,
    ) -> Pass:
        """Compute the `fresh` tokens of a sequence, reading `cached` cells for the others.

        tokens, pos: the whole sequence in order, with logical positions.
        cached: cells for the non-fresh tokens, in sequence order. Their keys stay
            rotated at their own `enc_pos`; `relocate` picks how the offset to `pos`
            is absorbed: "query" rotates one copy of the query per distinct offset,
            "rewrite" re-rotates the cached keys.
        groups: optional map from sequence index to a group id. When given, the
            head-averaged attention mass each fresh token puts on each group is
            returned per layer.
        depth: stop after this many layers. Cached cells are then only read up to
            that depth and the returned cells and hidden state are partial.
        """
        if relocate not in ("query", "rewrite"):
            raise ValueError(f"unknown relocate mode {relocate!r}")
        n = tokens.numel()
        if pos.numel() != n:
            raise ValueError("tokens and pos must have the same length")
        fresh = torch.ones(n, dtype=torch.bool) if fresh is None else fresh.bool()
        rows, held = fresh.nonzero().squeeze(1), (~fresh).nonzero().squeeze(1)
        m, n_held = rows.numel(), held.numel()
        if n_held != (0 if cached is None else len(cached)):
            raise ValueError("cached cells must cover exactly the non-fresh tokens")
        if n_held and not torch.equal(cached.tokens, tokens[held]):
            raise ValueError("cached cells do not hold the tokens of the sequence")

        dev, G, r, d = self.device, self.G, self.H // self.G, self.d_head
        pos_fresh = pos[rows]
        shift = torch.zeros(n, dtype=torch.long)
        if n_held:
            shift[held] = pos[held] - cached.enc_pos
        if relocate == "rewrite":
            offsets = [(0, torch.arange(n, device=dev))]
        else:
            offsets = [(int(s), (shift == s).nonzero().squeeze(1).to(dev)) for s in shift.unique()]

        rows_dev, held_dev = rows.to(dev), held.to(dev)
        visible = torch.arange(n, device=dev)[None, :] <= rows_dev[:, None]
        h = self.embed(tokens[rows].to(dev))
        key_phase = phases(pos_fresh, self.inv_freq, h)
        query_phase = [phases(pos_fresh - offset, self.inv_freq, h) for offset, _ in offsets]
        held_phase = phases(shift[held], self.inv_freq, h) if n_held and relocate == "rewrite" else None

        layers = self.layers if depth is None else self.layers[:depth]
        K_new, V_new = h.new_empty(len(layers), G, m, d), h.new_empty(len(layers), G, m, d)
        n_groups = 0 if groups is None else int(groups.max()) + 1
        mass = None if groups is None else torch.zeros(len(layers), m, n_groups)
        kept_q = h.new_empty(len(layers), self.H, m, d) if keep_queries else None
        kept_lse = h.new_empty(len(layers), self.H, m) if keep_queries else None
        groups_dev = None if groups is None else groups.to(dev)
        step = max(1, ATTENTION_FLOAT_BUDGET // max(1, self.H * n))
        narrow = h.dtype in (torch.float16, torch.bfloat16)

        for layer_index, layer in enumerate(layers):
            attn = layer.self_attn
            x = self._norm(layer.input_layernorm, h)
            q = attn.q_proj(x).view(m, self.H, d).transpose(0, 1)
            k = spin(attn.k_proj(x).view(m, G, d).transpose(0, 1), key_phase)
            v = attn.v_proj(x).view(m, G, d).transpose(0, 1)
            K_new[layer_index], V_new[layer_index] = k, v
            if kept_q is not None:
                kept_q[layer_index] = spin(q, key_phase)

            K_all, V_all = h.new_empty(G, n, d), h.new_empty(G, n, d)
            K_all[:, rows_dev], V_all[:, rows_dev] = k, v
            if n_held:
                held_keys = cached.K[layer_index]
                if held_phase is not None:
                    held_keys = spin(held_keys, held_phase)
                K_all[:, held_dev], V_all[:, held_dev] = held_keys, cached.V[layer_index]

            mixed = h.new_empty(m, self.H * d)
            wide = narrow and self.wide_scores
            K_read = K_all.float() if wide else K_all
            V_read = V_all.float() if wide else V_all
            for start in range(0, m, step):
                block = slice(start, start + step)
                size = min(step, m - start)
                scores = torch.empty(G, r, size, n, dtype=torch.float32, device=dev) if wide else h.new_empty(G, r, size, n)
                for (_, columns), (cos, sin) in zip(offsets, query_phase):
                    q_at = spin(q[:, block], (cos[block], sin[block])).reshape(G, r, size, d)
                    scores[..., columns] = (q_at.float() if wide else q_at) @ K_read[:, None, columns].transpose(-1, -2)
                scores = (scores * d**-0.5).masked_fill(~visible[block], float("-inf"))
                if wide:
                    weights = torch.softmax(scores, dim=-1)
                elif narrow:
                    weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
                else:
                    weights = torch.softmax(scores, dim=-1)
                if kept_lse is not None:
                    kept_lse[layer_index, :, block] = torch.logsumexp(scores, dim=-1).reshape(self.H, size).to(kept_lse.dtype)
                mixed[block] = (weights @ V_read[:, None]).permute(2, 0, 1, 3).reshape(size, self.H * d).to(mixed.dtype)
                if mass is not None:
                    per_column = weights.mean(dim=(0, 1))
                    per_group = per_column.new_zeros(size, n_groups).index_add_(1, groups_dev, per_column)
                    mass[layer_index, block] = per_group.float().cpu()

            h = h + attn.o_proj(mixed)
            h = h + layer.mlp(self._norm(layer.post_attention_layernorm, h))

        return Pass(Cells(tokens[rows].clone(), pos_fresh.clone(), K_new, V_new), h, mass, kept_q, kept_lse)
