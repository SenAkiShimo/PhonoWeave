# First author-attested DiffSinger teacher reference (v1.3)

This milestone does **not** convert missing reproducibility metadata into invented model identifiers.

The author confirmed `test2.wav` is a fresh direct LynneQing DiffSinger render of `mang_D4/phrase_a` from the supplied frozen MIDI, not a splice of the CVVC bank. This is **operator attestation**; origin is not cryptographically proven from WAV content. The exact model checkpoint and inference-project/settings are **not provided**.

The untouched PCM16 mono 44.1kHz input was validated against the six frozen note targets (all observed F0 errors <3 cents). Source duration is 2981ms vs 2870ms for the fixed score/output. The extra 111ms is a short decaying release tail; it is allowed only when a concrete decay gate passes, and comparisons remain fixed to the preview's shared timeline. This is *not* a per-policy optimization or per-policy alignment fit.

An explicit `author_attested_partial_provenance=true` declaration may now supply a **provisional** reference for descriptive, matched-budget K=2 comparisons. An unconfirmed source, altered MIDI, missing operator verification, copied preview, non-decaying tail, or unsupported source-metadata state still fails closed. The stricter `reference_verified_slots` only counts reference WAVs with both the required checkpoint identifier and project/settings reference. It remains **zero** for this sample, while `reference_provisional_author_attested_slots` is **one**.

On the actual user-provided sample, all 56 existing preview checksums and the frozen score are verified; 14 target slots exist, 1 is eligible for descriptive comparison and 13 are absent. At the fixed zero-shift alignment and identical 2140ms K=2 donor budget, the 12-band framewise log10 spectral-shape RMSE is:

| K=2 strategy | Spectral-shape error |
|---|---:|
| preserve i + i0 | 0.940079 |
| preserve ir + i0 | 0.956154 |
| preserve i + ir | 0.968741 |

These numbers **are not a sound-quality ranking**: the differences are small, alignment sensitive, based on a single short teacher sample, and the renderer is Rubber Band + crossfade rather than OpenUtau resampler/wavtool. The same target VC contexts are already present in the source, so unseen-phrase generalization has not been demonstrated. `split_share_efficacy_established` remains `false`.

The WAV itself, original source ZIPs and private operator receipt remain outside the GitHub repository. The author has a separate private archive containing the exact audio and receipt. Synthetic tests cover explicit partial provenance, no-attestation failure, and matched K=2 isolation.
