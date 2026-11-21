from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health_reports_the_held_out_set():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["calls"] == 449
    assert body["model"] == "late_pre_text_ac"


def test_calls_can_be_filtered_by_ticker():
    response = client.get("/calls", params={"ticker": "PEP"})
    assert response.status_code == 200
    calls = response.json()
    assert calls
    assert all(call["ticker"] == "PEP" for call in calls)
    assert calls == sorted(calls, key=lambda call: call["risk"], reverse=True)


def test_call_detail_includes_drivers_and_excerpt():
    listing = client.get("/calls", params={"limit": 1}).json()
    call_id = listing[0]["call_id"]

    response = client.get(f"/calls/{call_id}")
    assert response.status_code == 200
    detail = response.json()
    assert detail["call_id"] == call_id
    assert 1 <= len(detail["drivers"]) <= 6
    assert detail["transcript_excerpt"]


def test_unknown_call_returns_404():
    assert client.get("/calls/does-not-exist").status_code == 404


def test_metrics_are_sorted_by_validation_auc():
    response = client.get("/metrics")
    assert response.status_code == 200
    rows = response.json()
    assert rows[0]["model"] == "late_pre_text_ac"
    assert all(0 <= row["auc"] <= 1 for row in rows)
    assert rows == sorted(rows, key=lambda row: row["val_auc"], reverse=True)
