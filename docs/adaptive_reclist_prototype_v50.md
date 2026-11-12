# PhonoWeave v5.0 — first generated personal CV recording scheme

**Actual deliverable:** a CSV, syllable-only text file and split-hypothesis sheet, not another acoustic report. The user-supplied v4.2 cross-bank-proxy dataset produces a **31-CV coverage core + 7 additional context-probe CVs = 38 CV entries** spanning **20 author C and 31 author V labels**, with **11 tentative conditional-realization hypotheses**.

```bash
python -m phonoweave.adaptive_reclist_prototype_v50 \
  --proxy /path/to/cross_bank_proxy.json \
  --output /path/to/new_output_directory
```

Files: `recording_plan.csv`, `recording_tokens.txt`, `phoneme_split_proposals.csv`, `scheme.json`, and `README_CN.md`. Existing CV sound samples can be reused; recording these for a new singer is a pilot coverage task. **Do not treat the 38 CV entries as a complete production-ready CVVC/VCV singing voicebank.** It does not yet include every syllable, VC transition, CVC closure, breath, pitch range or articulation style.

### How computed

- Inputs are the **existing CV-OTO-timed** C/V spectral context proxy from v4.2. Existing VC aliases or counts play **no role** in choosing allophone groups. VC audio is retained pending proper rendering ablation.
- Initial K=2 context partition is accepted as a *pilot hypothesis* only when the correlated cross-bank proxy has >=10% K1-relative MSE improvement, >=3 of 4 held-out pitch banks improve, mean adjusted Rand index >=0.7 across refitted per-bank training partitions, and each context group has at least 2 attested contexts.
- K=3 is considered only if it improves the same relative proxy metric by at least six more percentage points over K=2 and still meets stability and minimum-context constraints. The chosen context groups are the modal partition across heldout folds.
- Build an observed CV graph from context membership in both C-side and V-side analyses; a graph edge must be mutually attested. Solve binary linear set cover to minimize distinct CV entries that cover each C and V label (the core) and then add as few extra CV entries as required to cover every *hypothesis-specific* context group. The core is pinned in the second optimization for an intelligible addition-only workflow.
- Every proposed subtype remains *unverified*. This is a coverage optimizer for **experimental sampling**, not a statistical estimate of universal allophone count nor optimal real singing synthesis quality.

### Honest boundaries

Cross-bank scores are **transductive**, from correlated pitch banks of one singer, not independent utterances. The data can produce an apparent orthographic contrast (notably i/u/uan/un) that primarily reflects ordinary language-specific phonetics; these are NOT automatically novel singer-accent-driven splits. Results assume upstream CV/OTO phoneme labeling is correct. Existing voicebank source audio/OTO are not changed, and no VC is deleted. A real inventory decision needs appropriate new recordings or fair source-budget-matched actual resynthesis and listening.

### Tests

```bash
python -m pytest -q tests/test_adaptive_reclist_prototype_v50.py
```

The stable full-data fixture from the user's supplied v4.2 report is **not committed** because it may contain private voicebank context; generic tests use synthetic proxy data.