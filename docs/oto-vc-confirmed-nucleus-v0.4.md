# Confirmed nucleus labels for LynneQingD4 (v0.4)

The voicebank author confirmed that `i` names the `li` nucleus, `ir` the
`ri`/`shi` nucleus, and `i0` the `zi`/`ci` nucleus. These are **alias semantics
for this specific voicebank**, not universal IPA standards. In particular,
multiple pronunciations under the same class are not claimed acoustically
identical. The original OTO and alias strings are unchanged.

## Run

After `phonoweave audit-oto-vc-zip` (PR #63):

```bash
phonoweave review-verified-vc-nuclei \
  --evidence /path/to/vc_oto_evidence.json \
  --confirmed-labels examples/lynneqing_d4_confirmed_labels.json \
  --output /outside/new_review_folder
```

The stored `source_oto_sha256` must match the input evidence. This command also
supports a compatible earlier local v0.1 probe containing
`provisional_candidates` and `spectrum_status` instead of `candidates` and
`spectral_status`.

The report enumerates each right-side surface label together with all three
left nucleus classes. It distinguishes **three independently relevant gates**:

1. Does the OTO list contain all three alias variants?
2. Does every variant have a measurable provisional window?
3. Can all three measurable windows be found in the same source WAV?

None proves the right-side phone identity, that an OTO window actually spans
a phonetic VC transition, or that splitting benefits synthesis.

## Real sample inventory (no audio added to GitHub)

Author-supplied LynneQingD4.zip initial probe has 131 VC-shaped entries whose
left alias is one of `i`, `ir`, `i0`, distributed over 45 different right-side
surface labels. Of these, 41 are complete triples, 19 triples have all three
measurable provisional 16ms windows, 20 complete triples share one source WAV,
and 10 triples are both measurable and share a WAV: `ai`, `ao`, `e`, `ei`,
`en`, `m`, `ou`, `sw`, `w`, `xw`. Only `m` is unambiguously a consonant in
this priority list without further alias convention review; other right-side
labels may represent vowels or glides. The module therefore avoids calling
all such groups phonetically verified VC transitions.

## Next steps

Review right-side alias phones and roles, and manually inspect the actual
transition boundaries. Attach measurement provenance and per-source-audio
quality. Avoid treating same-WAV segments or duplicate OTO rows as independent
replicates. A proper matched-cost study needs separately recorded takes and
held-out *rendered* evaluation before split/share decisions can be used.

Outputs are read-only, non-overwriting JSON and CSV. No production alias/OTO
edits or automatic sharing/deletion are authorized.

Run `pytest -q tests/test_vc_nucleus_review.py`.
