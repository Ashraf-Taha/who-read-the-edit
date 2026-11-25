## 3. Moving cached keys without writing them

After an edit of net length $\Delta$, every cell behind it has to behave as if it sat $\Delta$ positions away from where it was encoded. Published systems do this in one of two ways. Many rewrite the stored keys, multiplying each by $R(\Delta)$ [@yao2025cacheblend; @llamacpp3228; @he2025pie; @ma2026irminsul; @ma2026leyline; @ma2026kamera; @liu2026redknot; @ye2025kvcomm]. That writes every key behind the edit in every layer, ties a physical page to one offset, and rounds the keys again in the storage format. Others store keys without rotation and rotate them as they are read [@yang2025kvlink; @wang2025mepic; @ordonez2026minipic; @agarwal2025cachecraft; @xiao2024streamingllm]. That changes the cache format and, as an SGLang design discussion puts it, "touches every attention backend" [@sglang30928]. A third way keeps the standard format and writes nothing. Define the **offset** of a cached cell as $\delta_j = \pi_j - p_j$, its position now minus the position at which its key was rotated.

**Proposition 1 (query-side relocation).** For a query at position $\pi_i$ and a cached cell with rotated key $\tilde k_j$ and offset $\delta_j$, $a_{ij} = \big(R(\pi_i - \delta_j)\,q_i\big)^{\top}\,\tilde k_j$.

*Proof.* $\big(R(\pi_i-\delta_j)q_i\big)^{\top}R(p_j)k_j = q_i^{\top}R(\delta_j-\pi_i+p_j)k_j = q_i^{\top}R(\pi_j-\pi_i)k_j$, since $\delta_j+p_j=\pi_j$. $\square$

So the logits against all cells that share one offset come from rotating the query once, to position $\pi_i-\delta$, and reading the stored keys unchanged. With $D$ distinct offsets the query is rotated $D$ times and the partial logits are merged before the softmax. Dual Chunk Attention already uses this kernel pattern to score chunk-relative keys [@an2024dca]. The identity itself is a one-line consequence of RoPE being relative. Our contribution is to use it for relocating cached state after edits, an option the SGLang discussion of the other two does not consider [@sglang30928], together with the bookkeeping it needs. Table {{#relocation}} compares the three ways.

**Table {{#relocation}}.** Three ways to move cached cells to new positions.

| | Rewrite stored keys | Unrotated keys, rotate on read | Rotate the query per offset |
|---|---|---|---|
| Writes per relocation | every key behind the edit, every layer | none | none |
| Stored format and write path | standard | changed | standard |
| Extra work per attention call | none | one rotation per key read | one query rotation per distinct offset |
| One physical page usable at several offsets | no | yes | yes |
| Rounding accumulates over edits | yes | no | no |

**How many offsets.** Without repairs $D$ stays small. Each edit adds at most one run of cells with a new offset behind it, and an insert also adds its own cells as a run; since $D$ cannot exceed the number of runs, $D \le 1 + n_{\text{del}} + 2\,n_{\text{ins}}$ over the edits since the stale part was last re-encoded. Targeted repairs (Section 5) recompute scattered cells at offset zero, which splits runs but adds no new offset. Appendix A bounds both counts, and a sweep (Section 6) resets them. Per query the extra work is $D$ rotations of a $d$-dimensional vector per head, small next to attention over every cached key. In unfused PyTorch on the laptop GPU, scoring keys grouped by up to 32 offsets took {{e14.gather.min}} to {{e14.gather.max}} times as long as plain attention. Those calls are dominated by operation dispatch, so the ratio describes our implementation rather than a fused kernel (Appendix A).

**Exactness and drift.** In 64-bit floats, cells encoded 1,000 positions away from where they are used reproduce the re-prefill's logits to within {{e0.f64_relocated}}, by query rotation and by key rewriting alike. In 16-bit formats rewriting has a cost: when a cache is shifted again and again, the relative error of the rewritten keys grows with the square root of the number of rewrites (log-log slopes {{e0.slope_min}} to {{e0.slope_max}}), and after {{e0.max_edits}} rewrites in bfloat16 the output has moved by {{e0.kl_bf16_min}} to {{e0.kl_bf16_max}} nats, while query-side relocation stays at or below {{e0.kl_query_max}} (Appendix A). The drift is negligible for a handful of edits and visible after thousands.
