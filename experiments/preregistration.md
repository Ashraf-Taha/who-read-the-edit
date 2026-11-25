# Decision rules for the replication (E16)

The block below is copied without changes from the project's research log, from the entry timestamped 2026-11-23 13:39:14. It was written before the results of Qwen2.5-0.5B and SmolLM2-1.7B were examined; the results of Qwen2.5-1.5B had been seen. The hypotheses themselves are in the docstring of `experiments/scripts/e16_confirm.py`, whose SHA-256 begins with 36af5e09d34d2af0, the prefix recorded in the entry. That script is kept byte for byte as it was run, so its docstring still points to the research log itself, which is not part of this release.

---

**Decision rules for E16 (fixed now; the Qwen2.5-0.5B and SmolLM2-1.7B results have not been looked at; e16_confirm.py sha256 prefix 36af5e09d34d2af0).** α = 0.05, two-sided, per model; no pooling across models; each model reported separately.
- H1 holds on a model iff both RRF and the bridge score have lower mean KL than reader attention alone at 10% and both Holm-adjusted (family of the two) Wilcoxon p < 0.05; otherwise not confirmed (fails if either is significantly worse).
- H2 holds on a model iff reader-first (found) has lower mean KL than RRF in the short-suffix stratum and higher in the long-suffix stratum, each with Wilcoxon p < 0.05; it fails if either stratum is significant in the opposite direction; otherwise not confirmed. (The earlier generated wording "direction holds, not significant in every stratum" is withdrawn: that case is "not confirmed".)
- H3 holds on a model iff, with the notes, the most-read-per-token rule finds the question in more than half of the cases, the shortest-segment rule in fewer than half, and found-segment-first has lower KL than shortest-first with p < 0.05; it fails if the attention rule finds the question in half of the cases or fewer.
- H4 holds on a model iff the stale state's excess log-probability for the deleted document's first sentence (over the re-prefill) has a 95% interval above zero and the bridge repair's excess is smaller than the stale state's; it fails if the stale excess interval includes zero or lies below it.
- Sensitivity: every test is also reported without the questions that appear in any earlier result file (HotpotQA ids), since one of the 36 Qwen2.5-1.5B questions was used before.
