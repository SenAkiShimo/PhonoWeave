# v0.3 — OTO VC evidence bridge (archived real voicebanks)

This module links **user-authored OTO surface-alias candidate intervals** to the
new language-neutral *phonetic transition graph*, but only after manual
identity mapping. It does not assume IPA identity from labels like \`i\`,
\`ir\`, \`i0\`, \`v\`, or \`D4\`.

## Usage

\`\`\`bash
phonoweave audit-oto-vc-zip LynneQingD4.zip \
  --prefix D4 --output /tmp/lynne_d4_vc_audit
\`\`\`

The output directory must not exist. It contains:

- \`vc_oto_evidence.json\`: source ZIP/OTO hashes, structural inventory and
  fixed-window spectral fingerprints for candidate aliases.
- \`vc_review_queue.csv\`: source WAV, original alias, 1-based OTO line number,
  raw OTO timing and provisional 16 ms window status.
- \`identity_mapping_template.csv\`: distinct aliases, grapheme-like left/right
  surface names, and empty phone identity fields for human review.

No archive extraction, source WAV/OTO changes, resynthesis, reassignment of
production aliases or automatic split/share decisions are performed.

## Initial LynneQingD4 analysis (local author-provided reference)

A user-provided D4 ZIP was inspected locally; the audio and entire archive
were **not** added to version control. It has 25 mono PCM16 44.1 kHz WAVs and
1,935 OTO records. 1,124 records have the expected two-token VC alias shape,
representing 1,007 distinct VC alias strings. The 16 ms OTO-relative
fingerprint window was measurable for 830 entries and too short for 294.
These counts are **acquisition inventory indicators**, not acoustic or
perceptual accuracy scores.

Example source identities in this sample:

| Existing alias | OTO line | Source WAV |
| --- | ---: | --- |
| \`D4i sh\` | 1637 | \`vcsection3.wav\` |
| \`D4ir sh\` | 922 | \`ingi0ir.wav\` |
| \`D4i0 sh\` | 1641 | \`vcsection3.wav\` |
| \`D4i m\` | 894 / 1529 | \`eriin.wav\` / \`vcsection2.wav\` |
| \`D4ir m\` | 1537 | \`vcsection2.wav\` |
| \`D4i0 m\` | 1535 | \`vcsection2.wav\` |

The metadata proves that these different surface labels exist, **not** their
IPA values, speaker-specific acoustic realization distinctions, phoneme
boundaries or a need for splitting.

## Integration contract

\`map_reviewed_graph_edges(oto, graph, reviews)\` accepts:

\`\`\`json
{
  "schema": "phonoweave.oto_vc_manual_identity_review.v0.1",
  "links": [
    {
      "alias": "D4ir sh",
      "left_surface": "ir",
      "right_surface": "sh",
      "left_phone": "REVIEWED_IPA_LEFT",
      "right_phone": "REVIEWED_IPA_RIGHT",
      "phase": "across_syllables",
      "pattern": "VC",
      "identity_reviewed": true
    }
  ]
}
\`\`\`

Only transitions already represented in the matching versioned phonetic
graph are linkable. Unmapped entries remain unresolved; these cannot
automatically authorize a production singer alias.

## Critical limitations

The chosen OTO region is \`[offset+overlap, offset+preutterance]\` and its
16 ms center is used for a coarse normalized log-spectrum. This region
may capture a vowel, consonant or silence rather than the actual boundary.
Only a human-checked transition window could support a subsequent
synthesis-relevance evaluation.

Repeat OTO alias rows are not necessarily independent acoustic takes.
Training/test separation, phonetic boundary review, pitch controls,
perceptual tests and measured recording-duration matching remain future
milestones.

Run \`pytest -q tests/test_oto_vc_evidence.py\` and the full test suite.
