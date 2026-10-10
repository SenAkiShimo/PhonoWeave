from phonoweave.calibration_negative_fullscan import cluster_events

def test_event_deduplication():
    assert cluster_events([100, 104, 109, 401, 409]) == [104.0, 405.0]

def test_no_events():
    assert cluster_events([]) == []
