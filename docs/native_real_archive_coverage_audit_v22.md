# PhonoWeave v2.2 — real archived coverage audit and conservative alias mapping

## Purpose

Prioritize the real product goal: **inspect a singer's sound inventory, avoid pointless rerecording, and output the smallest justified supplementary recording list**. This patch changes neither acoustic models nor the original voicebank.

## Real LynneQing frozen-source check

Read-only inspection of the previously preserved \`PhonoWeave_User_Native_Batch_Discovery_v2.zip\`, specifically \`results/lynneqing_seven_subbanks/source_index.csv\`. Explicit mapping for its seven banks:
\`mang_D4=D4\`, \`mang_D5=D5\`, \`mang_G3=G3\`, \`mang_G4=G4\`, \`lu_B4=B4\`, \`lu_C4=C4\`, \`lu_F4=F4\`.

- **7** subbanks belonging to the same singer.
- **175** indexed WAV source slots: **25 source WAV filenames per bank**, with all 25 names shared across the seven banks.
- **7,860** accepted OTO-relative acoustic windows; these are *not* all raw OTO entries.
- Explicitly removing the known pitch prefixes yields a **union of 1,007** author-defined alias strings and an **intersection of 1,006** in the *accepted-window source index*.
- The only missing item from that accepted-window index union is **\`ai a\` for \`lu_C4\`**. This does **not** prove a missing OTO alias, WAV or vocal performance: its feature extraction may simply have abstained. Do not add it to the reclist automatically.
- **82–84** normalized aliases per bank map to more than one source WAV path in this accepted-window index. These are genuine reasons to check source mapping, not automatically rerecord.

The source index contains no complete original voicebank WAV archives or full \`oto.ini\` files. It is inadequate for a ground-truth supplement precision/recall score, independent singing-performance counts, or a claim that the voicebank is fully complete.

## Small fix

The v2.0/v2.1 native planner only matched exact WAV filenames or exact OTO alias names. Long-WAV source banks may instead use aliases like \`D4a shw\` on \`a.wav\`, while an author-supplied target is just \`a shw\`.

A v1 manifest can now optionally declare **\`bank_alias_prefixes\`**. The planner strips only *the exact prefix specified by the author*, never a guessed pitch, language identifier or IPA transform. The old plain-text reclist remains exact-matching only.

\`\`\`json
{
  "schema": "phonoweave.user_native_recording_targets.v1",
  "bank_alias_prefixes": {"D4": "D4", "G3": "G3"},
  "targets": [
    {"line": "a shw", "wav": "a shw.wav", "subbanks": ["D4", "G3"]}
  ]
}
\`\`\`

For \`D4/oto.ini\` containing \`a.wav=D4a shw,...\`, and readable \`D4/a.wav\`, the author-declared mapping validates **existing audio coverage** without recommending a redundant recording. If the normalized alias is found under multiple WAV paths, it produces a *source/OTO review* status, not a recording instruction.

**Conservative fail-closed change:** If the target's specified WAV is absent but a bank contains other WAVs and no alias mapping matches, the result is \`unverified_template_mapping\`, not \`missing_audio\`. The singer may be using a long-WAV naming scheme; request a manifest mapping instead of asking them to rerecord. Empty bank with no WAV still allows a missing-audio recommendation. A known alias pointing to a missing/invalid WAV remains a direct missing/invalid source problem.

## Test and rollout boundary

Synthetic PCM tests cover declared prefixes, no-guess/no-prefix abstention, wrong prefixes, two alias paths for one target, multiple alias windows in the same long WAV, rejection of path/control characters, and compatibility with legacy text templates. These are not independent phonetic validation.

No production voicebank, source WAV, OTO or aliases are modified, and no phoneme identity is inferred. Keep the PR Draft until CI has actually completed and full original voicebank WAV/OTO can be audited.
