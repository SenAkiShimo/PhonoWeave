# PhonoWeave v9 — IPA-independent acoustic structure discovery

**Research prototype, not validated phone recognition.** This release implements two *automatic* analyses on archived real acoustic evidence. It needs **no new manual annotation** and does not change `oto.ini`, user symbols, voicebank audio, `speaker_profile` or the recording scheme. It does not claim new phonemes or validated synthesizable units.

## Research objective

Can repeated acoustically similar structures be extracted and screened without translating user symbols into IPA or assuming that one event equals one phone? What can be learned when only archived evidence is available, and which apparent findings disappear after controlling for bank/style/template effects?

## Engine A: source-linked 36D spectral context subtype probes

Module: `phonoweave/native_acoustic_subtypes_v9.py`

- Input: the **frozen v3 ZIP** containing 7,860 `late_context` 36D acoustic vectors and aligned user-chosen labels (23 labels, 175 source-bank WAV identities, 25 source-WAV filename templates).
- Crucial limitation: the original v3 vectors were extracted at OTO-derived reference locations. This engine clusters these **spectral-context samples**, not unanchored phone segments. It does NOT independently find candidate boundaries.
- Each user's exact symbol groups rows; k=2 clustering within each symbol uses only the spectral vectors. No IPA, phonological dictionary, user contexts, source names or bank IDs enter k-means distance. Bank IDs are used only for training-only mean correction and the posthoc confound/robustness audit; each style trains a model using only development templates.
- The v3 frozen dev/test recording templates remain unchanged. All training centering/scaling and k-means centroids use development templates only. A fixed style-consistency diagnostic uses Adjusted Rand Index (ARI) on heldout templates. Additional geometry, source-WAV, bank and context-confound guards restrict reports.
- **Methodological disclosure:** the frozen v3 heldout set was also used in v3 feature-mode evaluation, and the v9 confound guard was added after inspecting exploratory v9 pilot outcomes. Hence this is **retrospective discovery**; the heldout is *not* an untouched confirmatory test of a new hypothesis.
- This dataset contains seven subbanks, but does NOT establish seven independent speakers. They come from the same LynneQing material. Correlated scripts repeated at pitches or styles are not independent replication.

Run:

```bash
python -m phonoweave.native_acoustic_subtypes_v9 \
  --v3 /path/to/PhonoWeave_User_Native_Temporal_Context_v3.zip \
  --output /new/empty/v9_context_results
```

Results: `subtype_diagnostics.csv`, `sample_membership.csv` and `summary.json`. Membership IDs `o__ctxcluster_0` etc. are internal research IDs, **not recording aliases**.

### Actual frozen LynneQing result

- 7,860 actual spectral-context samples and 23 exact user-native symbols were analyzed.
- 22 symbols did not pass the exploratory geometry screen or had insufficient frozen-heldout evidence.
- `o` passed initial geometry with cross-style heldout ARI = **0.8014**, heldout one-vs-two centroid reconstruction gain **0.1051**, but **failed the stronger bank/template guard**: only **one distinct heldout source WAV template** and bank-association NMI **0.4303**. It is flagged `exploratory_confound_or_underreplicated`, not a validated new subtype.
- **Zero** symbols passed every guard. That is not evidence that all existing labels have only one realization.

## Engine B: directly extracting repeated spectral motifs from PCM WAV

Module: `phonoweave/native_wav_motifs_v9.py`

- Reads real WAV PCM16 audio from ZIP, not v3 embeddings. No OTO timings, source aliases, user pronunciation list, IPA or linguistic dictionary are fed to segmentation or clustering.
- Computes 25-ms Hann FFT and 24-band log-mel spectra every 10 ms. Derives activity spans and spectral-change maxima, producing **unverified acoustic intervals**. Builds 48D features from spectral shape and early/late changes and applies deterministic 2-centroid discovery.
- Uses source file paths **only for identity and grouping** (never acoustic features). Removes `_OTO...` suffix when grouping equivalent context clips across subbanks; this is not an alignment, duration, onset or phone label.
- Fixed dev/heldout grouping by an opaque clip-group hash; separately fits two style-specific models and compares heldout assignment consistency, but this is not an independent-speaker result.

Run:

```bash
python -m phonoweave.native_wav_motifs_v9 \
  --audio-zip /path/to/audio_context_archive.zip \
  --output /new/empty/v9_direct_wav_results
```

**Real archive demonstration:** the historical `PhonoWeave_Actual_Phone_Segmentation_v0.1.zip` includes 168 600-ms `source_context_600ms.wav` files, selected originally around OTO-linked `sh` experiments. Although the v9 waveform segmenter never reads those OTO times, the **archive selection itself is not unanchored**; these are not the 175 full-length original recordings, and no complete phone inventory can be inferred from them.

On those **168 audio clips**, Engine B found **503 short acoustic regions** and two broad unsupervised spectral motifs: sizes **336** and **167** across 23 clip groups. Median region lengths were **210 ms** and **110 ms** respectively. Cross-style heldout motif assignment ARI was **1.0** for 63 heldout regions, but **this can reflect different broad classes of acoustic material within the same context (e.g., frication versus voicing), not two phones or two realizations of the same phone**. Clips share a speaker/script and were originally selected using OTO; this does not validate phones or new phoneme labels.

Outputs: `motif_regions.json`, `raw_spectral_embeddings.npz`, `summary.json`, with source PCM SHA-256, archive SHA-256, local interval positions, and explicit unverified status. No raw WAV is exported or published.

## Why these two methods must remain separate

Engine A has useful source user-symbol groupings but relies on OTO-derived sample locations. Engine B can extract intervals and spectra directly from waveform without OTO timing but its historical archived **audio subset** is limited and preselected. **Neither source alone supports blind, independently validated original-full-WAV phone discovery.** In particular, overlaying A's labels on B's regions to create a newly 'recognized phone' without provenance and independent verification would reintroduce the same source leakage v5 was designed to avoid.

The next algorithm milestone should be `v10`: record-agnostic WAV extraction on actual full-length raw archives, repeatable cross-recording spectral motif association with explicit unknown class, and only then optional links to user-authored scripts and resynthesis A/B—not automatic alias splitting.

## Testing and provenance

- 17 deterministic unit tests for clustering, name opaqueness, source-bank normalization, single-template abstention, raw PCM, spectral discrimination, silence and incomplete audio evidence.
- Both real-data runs are included in the artifact package with source archive SHA-256.
- No external speech data required. Dependencies only numpy + Python stdlib for the new engines.
- Algorithms are heuristic research prototypes, not trained or validated across independent speakers.
