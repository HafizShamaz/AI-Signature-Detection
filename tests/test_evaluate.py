from training.evaluate import evaluate_candidate

def test_passes():
    assert evaluate_candidate({"map50":0.9,"precision":0.9})["approved"]

def test_fails():
    assert not evaluate_candidate({"map50":0.5,"precision":0.9})["approved"]
