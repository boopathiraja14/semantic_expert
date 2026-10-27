import json

import pytest

import kb


def test_level0_is_small_and_lists_datasets():
    l0 = kb.level0()
    assert set(l0["domains"]) == {"loss_events", "kri", "rcsa"}
    assert "kri_values" in l0["domains"]["kri"]["datasets"]
    text = json.dumps(l0)
    assert "fields" not in text and "$ref" not in text   # no L1/L2 detail


@pytest.mark.parametrize("question, expected", [
    ("when does a red kri go to the cro", "kri.kri_escalation"),
    ("what is a near miss", "loss_events.event_definition"),
    ("collateral documentation error credit or operational", "loss_events.boundary_events"),
    ("how is residual risk scored", "rcsa.risk_scoring"),
    ("which dates do reporting deadlines use", "loss_events.key_dates"),
    ("internal fraud", "loss_events.basel_event_types"),
])
def test_bm25_top_hit(question, expected):
    assert kb.search_sections(question)[0]["path"] == expected


def test_search_returns_only_top_sections_and_respects_domain():
    hits = kb.search_sections("threshold escalation", domain="kri")
    assert 0 < len(hits) <= 3 and all(h["path"].startswith("kri.") for h in hits)
    assert kb.search_sections("zzzz") == []


def test_refs_expanded_inside_sections():
    rag = kb.search_sections("rag amber green", domain="kri")[0]["content"]["rag_status"]
    assert rag["red"].startswith("Outside appetite")
    for s in kb._sections():
        assert "$ref" not in json.dumps(s["content"]), s["path"]


def test_circular_ref_raises(monkeypatch):
    monkeypatch.setattr(kb, "read_yaml", lambda rel: {"a": {"$ref": "x.yaml#a"}})
    with pytest.raises(ValueError, match="circular"):
        kb.expand_refs({"$ref": "x.yaml#a"})


def test_read_outside_knowledge_folder_rejected():
    with pytest.raises(ValueError):
        kb.read_yaml("../kb.py")


def test_load_tables_only_requested_datasets():
    t = kb.load_tables("oprisk_losses", ["op_loss_events"])
    assert [d["name"] for d in t["datasets"]] == ["op_loss_events"]
    assert "gross_loss_amount" in [f["name"] for f in t["datasets"][0]["fields"]]
    assert [r["name"] for r in t["relationships"]] == ["recoveries_to_events"]  # join path shown
    names = [m["name"] for m in t["metrics"]]
    assert "gross_loss_amount" in names and "recovery_amount" not in names


def test_load_tables_errors_list_options():
    with pytest.raises(KeyError, match="oprisk_kri"):
        kb.load_tables("nope", ["x"])
    with pytest.raises(KeyError, match="kri_values"):
        kb.load_tables("oprisk_kri", ["nope"])


def test_every_section_table_link_resolves():
    for s in kb._sections():
        for link in s["tables"]:
            model, dataset = link.split(".")
            kb.load_tables(model, [dataset])  # raises if a link is broken
