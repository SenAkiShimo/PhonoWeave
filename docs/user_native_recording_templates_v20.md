# PhonoWeave v20 — User-native recording templates

PhonoWeave's core goal remains speaker-specific **recording inventory planning**, not building a singing renderer or proliferating audio clustering models. The v16–v19 recording planner uses Mandarin legal-syllable generation. This optional v20 workflow checks **author-existing recording templates** with their original opaque Unicode names, avoiding Mandarin or IPA normalization.

## Usage

Existing text reclist (one original recorded line per row):

\`\`\`bash
phonoweave plan-recording /path/to/voicebank \
  --native-reclist /path/to/author_original_reclist.txt \
  --max-lines 8
\`\`\`

For a specific pitch bank, repeat \`--native-bank A3\`. The optional \`--existing-reclist\` marks items already scheduled but **not actually recorded**. Use \`--export /new/dir\` for a new, out-of-voicebank folder containing \`supplement_reclist.txt\` and \`recording_plan.json\`.

The existing local GUI's **Recording Plan / 补录方案** screen also accepts an optional path to the original recording template. Leave blank for unchanged Mandarin analysis.

A plain-text line named \`ŋa\` is expected in a file named \`ŋa.wav\` in each selected subbank, or linked by an exact OTO alias \`ŋa\`. When original script lines and source WAV names differ, use a JSON target mapping:

\`\`\`json
{
  "schema": "phonoweave.user_native_recording_targets.v1",
  "targets": [
    {"line": "ŋa", "wav": "source-take-01.wav"},
    {"line": "tɕi", "wav": "tɕi.wav", "subbanks": ["A3", "D4"]}
  ]
}
\`\`\`

A mapping and its original names are **user-supplied evidence**, not the output of automatic phoneme segmentation. Without a known source template, PhonoWeave must **abstain** rather than invent a cross-product of unverified sound sequences. An OTO subbank directory is only a recorded pitch/style configuration, never an independent speaker.

## Evidence semantics

- \`missing_audio\` and \`invalid_audio\`: candidate existing-template recording task. The output lists the specific subbanks affected.
- \`audio_present_needs_oto_review\`: recorded WAV exists but is not referenced by OTO; do not request an unnecessary rerecording.
- \`ambiguous_file_names\` or \`ambiguous_oto_alias\`: source identity unresolved, request investigation.
- \`audio_present_registered\`: WAV and OTO both exist; **not** proof of correct pronunciation, timing, independent takes or synthesis coverage.
- Identical PCM detected across subbanks is flagged. Two distinct PCM files are not proof of independent vocal performances.
- Scheduled but not recorded items are skipped in new tasks, but remain missing from observed audio coverage.
- Budget counts **distinct recording-list lines**, not total takes across subbanks.

The planner leaves original WAVs, OTO, alias conventions, synthesis inventories and the default Mandarin pathway unmodified. This release does not automatically add newly discovered phones or claim that all language-independent recording-scheme decisions have been solved.

## Validation

Synthetic, genuine PCM16 WAV fixtures exercise non-ASCII original labels, missing audio in one pitch subbank, copied PCM, invalid PCM, exact alias mapping, optional target JSON, scheduled-but-unrecorded lines, zero budget, path validation and legacy CLI/GUI route presence. This is software integration evidence, **not** speaker-generalized acoustics, true phone-boundary detection, or a measured improvement in actual recording hours.

Next step should evaluate this exact-template and source-aware workflow on a complete independently provided voicebank / recording script, and then connect user-native **context coverage** only when there is an actual source-grounded list of possible realizations. Do not automatically generate unattested phoneme combinations.
