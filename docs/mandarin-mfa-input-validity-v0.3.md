# Mandarin MFA input validity gate v0.3

This is an **input-validity correction**, not a claim that Mandarin phone segmentation is solved.

The real four-clip LynneQing pilot aligned under MFA 3.4.3, but bare `a` was mapped to `tɕ e˥ n`; `rong` to `n u˩ ŋ`; and two `ma` intervals covered only 11.45% and 44.46% of recorded squared PCM energy. Re-running with `--no_tokenization` produced byte-identical TextGrids. None may be used as accepted phone boundaries.

The original source positions, OTO rows, ZIP hashes, and PCM extraction of all four samples have now been verified. OTO correspondence is **not** proof of a source-author-verified complete transcript. A separate, author-confirmed six-syllable DiffSinger render `li ma shi sha zi ma` provides a complete-sequence control, though its lexical tones and phone boundaries are not independently labeled.

The private five-sample package supplies two parallel orthographic *proxies* (chosen Hanzi and tone-marked Pinyin) over **identical PCM**, with source provenance. Chosen characters and lexical tones are not asserted to be the actual words or sung lexical tone. This module checks for implausible phone sequences, low word-energy coverage and invalid time intervals. Even a provisional pass requires human boundary review.

Invoke with `PHONOWEAVE_MFA_PACK_DIR=/path/to/private_pack python -m phonoweave.mfa_input_quality_gate` after executing MFA. The private package contains the one-command `run_once.sh` wrapper; **no private WAVs or author reference are committed to GitHub**.

Per the MFA 3.4.3 `align_hf` source, `--no_tokenization` selects language=`unknown`, and `--use_g2p` supplies pronunciations for OOV items; neither overrides all dictionary pronunciations.

Documentation: https://huggingface.co/MontrealCorpusTools/mandarin_mfa and https://huggingface.co/MontrealCorpusTools/mandarin_mfa/blob/main/acoustic/phones.txt

**Gold-quality phone boundaries in the actual pilot: 0.**
