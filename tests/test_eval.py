"""Judge aggregation: verdict parsing, resolve/combine, and that the position
swap cancels a position-biased judge."""
import json

from lmkit.eval.judge import combine, parse_verdict, resolve, run


def test_parse_verdict_marker():
    assert parse_verdict("blah\nVERDICT: A") == "A"
    assert parse_verdict("VERDICT: B") == "B"
    assert parse_verdict("VERDICT: TIE") == "TIE"


def test_parse_verdict_last_marker_wins():
    assert parse_verdict("first I lean A ... VERDICT: B") == "B"


def test_parse_verdict_reasoning_then_marker():
    assert parse_verdict("A is ok but B follows the instruction better.\nVERDICT: B") == "B"


def test_parse_verdict_ambiguous_is_tie():
    assert parse_verdict("not sure") == "TIE"
    assert parse_verdict("") == "TIE"


def test_resolve_and_combine():
    assert resolve("A", "m1", "m2") == "m1"
    assert resolve("B", "m1", "m2") == "m2"
    assert resolve("TIE", "m1", "m2") == "tie"
    assert combine("x", "x") == "x"
    assert combine("x", "y") == "tie"
    assert combine("tie", "tie") == "tie"


def test_run_position_swap_credits_consistent_winner(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text(json.dumps({"id": "1", "messages": [{"role": "user", "content": "q"}],
                             "completion": "good"}) + "\n")
    b.write_text(json.dumps({"id": "1", "messages": [{"role": "user", "content": "q"}],
                             "completion": "bad"}) + "\n")

    def jf(messages, endpoint, model):  # picks whichever slot holds "good"
        content = messages[1]["content"]
        slot_a = content[content.find("Response A:"):content.find("Response B:")]
        return "VERDICT: A" if "good" in slot_a else "VERDICT: B"

    rep = run(str(a), str(b), "A-model", "B-model", judge_fn=jf)
    assert rep["wins_a"] == 1 and rep["wins_b"] == 0 and rep["ties"] == 0
    assert rep["win_rate_a"] == 1.0
