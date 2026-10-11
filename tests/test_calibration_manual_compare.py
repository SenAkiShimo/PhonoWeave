from phonoweave.calibration_manual_compare import compare_predictions


def test_join_json_and_missing_predictions() -> None:
    labels = {"labels": {
        "12": {"prompt_index":12,"base_unit":"x","class_name":"fricative","context_family":"plain",
               "occurrences":{"1":{"anchor_ms_after_cue":120,"status":"ok"},
                              "2":{"anchor_ms_after_cue":200,"status":"ok"}}},
        "15": {"prompt_index":15,"base_unit":"sh","class_name":"fricative","context_family":"rounded",
               "occurrences":{"1":{"anchor_ms_after_cue":250,"status":"uncertain"},
                              "2":{"anchor_ms_after_cue":300,"status":"ok","outside_expected_slot":True}}}
    }}
    paired = {"anchors":[{"prompt_index":12,"qc":"usable",
                          "occurrence_1_anchor_ms_after_cue":130,
                          "occurrence_2_anchor_ms_after_cue":180}]}
    sibilant={"rows":[{"prompt_index":12,"occurrence":1,"onset_ms_after_cue":None,
                       "status":"unresolved","reason":"no_sustained_aperiodic_frication"},
                      {"prompt_index":12,"occurrence":2,"onset_ms_after_cue":195,
                       "status":"review","reason":"repeat_lag_disagreement"}]}
    r=compare_predictions(labels,paired,sibilant)
    assert r["manual_annotations"]["eligible"]==2
    assert r["manual_annotations"]["uncertain"]==1
    assert r["manual_annotations"]["out_of_slot_review"]==1
    a=r["models"]["paired_v0.3.3_all"]["summary"]
    assert a["matched"]==2 and a["mae_ms"]==15
    assert r["models"]["sibilant_v0.6_all"]["summary"]["missing"]==1
    assert r["models"]["sibilant_v0.6_candidate"]["summary"]["matched"]==0
    assert r["sibilant_unresolved_reasons"]["no_sustained_aperiodic_frication"]==1
