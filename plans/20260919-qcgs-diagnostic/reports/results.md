# QCGS diagnostic — validated CUDA results, 2026-09-19

**Decision: STOP — ordered gate 4, for the frozen single-view entropy-sign signal.**
This replaces the earlier INCONCLUSIVE status caused by unavailable local CUDA.
The supplied Kaggle artifact completes the diagnostic; it does not establish
that all label-free support-selection signals are impossible.

All ten completed runs passed offline validation: two CUDA smoke cells, four
outcome-blind scans, two Stage A cells and two Stage B cells. Source/config,
file hashes, parameter order, registry replay, dedicated random draws, candidate
coverage and raw utility arithmetic matched. Recomputed analysis JSON equals
the archived analysis, including uncertainty/strata; independent CE subtraction
and `math.fsum` also reproduce the primary means and STOP predicate.
The Kaggle CPU prerequisite suite passed 404 tests with six hardware skips;
CUDA smoke separately scored two real-model ID queries per cell.

Stage A has 16 queries/cell and verifies all 5,090 / 10,620 legal swaps.
**Exact oracle headroom is positive for 16/16 queries in each cell**, with mean
CE gains **+0.408288 / +0.120676**. Entropy-sign mean per-query Spearman rho is
only **0.018160 / 0.057525** (16/16 defined per cell; positive rho in 9/16 and
10/16). Entropy-cosine mean rho is −0.009202 / −0.003933. These are query-level
correlations over the complete legal spaces, not pooled swap observations.

Stage B confirms the frozen policies on 128 new ID queries/cell. Positive
utility means improvement; `U = CE_Ramen − CE_entropy-sign` and
`D = CE_random − CE_entropy-sign`.

| Stage B measure | OOD=0 | OOD=0.5 |
|---|---:|---:|
| Mean U, primary versus Ramen | −0.018072 | −0.014818 |
| Mean D, primary versus random | −0.013258 | −0.011178 |
| Mean entropy reduction | +0.031066 | +0.025518 |
| Ramen → entropy-sign correct queries | 65/128 → 64/128 | 57/128 → 57/128 |
| Helpful / harmful replacements | 70 / 58 | 67 / 61 |
| Actual replacements / no-swaps | 128 / 0 | 128 / 0 |
| Lower entropy but worse CE | 27/128 | 36/128 |
| 10% symmetric trimmed mean U / D | −0.009061 / −0.004174 | −0.010142 / −0.006704 |
| Mean U / D after dropping two largest each | −0.026406 / −0.022499 | −0.024107 / −0.020355 |

Control mean CE utility (OOD=0 / 0.5): random −0.004815 / −0.003640;
entropy-cosine −0.006854 / −0.010320; supervised first-order reference
+0.174322 / +0.413015. Primary utility medians are slightly positive
(+0.000055 / +0.000051), but harmful swaps are larger on average: mean gain
among helpful queries +0.061075 / +0.081521 versus mean loss among harmful
queries −0.113595 / −0.120632. Lower average entropy therefore does not imply
better supervised utility. No score, threshold, bins or sample selection was
changed using Stage B outcomes.

Gate order: no detected integrity failure → complete quotas → GO_PILOT fails
→ **STOP holds because mean U≤0 and mean D≤0 in both cells**. STOP precedes
REVISE even though the entropy/CE mismatch condition also holds. Do not advance
this signal to a one-swap pilot, full matrix, reranking or adaptive support size.

The descriptive block intervals include zero: with 5 / 6 scored blocks, U 95%
intervals are [−0.048440, +0.014216] / [−0.036285, +0.010141], and D intervals
are [−0.042993, +0.013294] / [−0.031832, +0.017461]. These conditional one-stream
intervals do not change the preregistered STOP gate or establish statistical
proof of harm. This is one seed on CIFAR-100-C, with shared cache history.

Registry checks exclude 236 historical base-image IDs; all four smoke IDs
were already in that exclusion set. There are no duplicate base images within
a cell and no Stage A/B overlap across either OOD cell. Cross-OOD overlap within
the same stage is permitted and observed: two Stage A IDs and 15 Stage B IDs.
Scans expanded from 600 to 1,200 rows solely for unseen-query quotas, then the
scored prefixes were frozen at 400 / 700 rows. Input ledger commit `18e98fa`
precedes Stage A; `7367f2b` freezes the Stage A bins before Stage B.

Stage B verifies 459 / 475 distinct selected actions. Its mean best-verified
utility is +0.189193 / +0.425794, a **lower bound**, not an exact oracle.
Exact oracle utility/regret fields remain null there. Measured Stage B pool/
aggregate/scoring latency averages 94.1 / 184.2 ms/query, plus shared retrieval;
scoring peak extra allocation averages 226 / 493 MiB versus about 7.24 GiB for
evaluator work. These diagnostic measurements include bookkeeping and do not
benchmark a finished deployable selector. Full metrics, transitions, regrets,
block counts, leave-one-block-out ranges, strata and costs are in the analysis.

Evidence and provenance (large artifacts remain outside Git):

- [Raw query records](../../../evidence/qcgs-kaggle-b7659601e54c/qcgs-label-free-evidence/queries.jsonl),
  [recomputed analysis](../../../evidence/qcgs-kaggle-b7659601e54c/qcgs-label-free-evidence/analysis.json),
  [validation receipt](../../../evidence/qcgs-kaggle-b7659601e54c/validation.json),
  [frozen provenance](../../../evidence/qcgs-kaggle-b7659601e54c/qcgs-label-free-evidence/locks/preflight.json).
- Original ZIP: `qcgs-label-free-evidence.zip`, SHA-256
  `b7659601e54c3862bd31b4f6917279c2326054ef98884fbf3bfdc6aef51c317c`.
- Source `8298ba434fa3e63710b899500a822a3575ca4884`, clean at execution;
  Tesla T4, Python 3.11.16, Torch 2.4.1+cu121. Registry SHA-256
  `e173046e760cce92e1015a8020316d3c140722275e8aaffcc943935100686f93`.
- Canonical recomputed analysis SHA-256
  `eeb8f0c2df01ec208593025e4b08487333ff80553793614ed5384e0660ecf813`.

This audit checks archived provenance and raw arithmetic. It did not redownload
and rehash the original dataset/model or independently rerun CUDA logits.
The original notebook/source pin and scientific protocol are retained; only
post-run status and this report are updated.
