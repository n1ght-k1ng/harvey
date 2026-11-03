"""/api/funnel: contacts at each stage, cumulative, and why the rest stopped."""

import asyncio
import sqlite3

from tests.test_dashboard_signals import client  # noqa: F401  (fixture)
from harvey.dashboard import _funnel_reach
from harvey.state import StateManager


def _sql(db, query, params=()):
    with sqlite3.connect(db) as conn:
        conn.execute(query, params)
    conn.close()


def _prospect(db, pid, status="new", email_status="", company="c1", created="2026-01-01T10:00:00"):
    _sql(db, "INSERT INTO prospects (id, company_id, first_name, email, status, email_status, created_at) "
         "VALUES (?, ?, 'Pat', ?, ?, ?, ?)", (pid, company, pid + "@example.com", status, email_status, created))


def _outbox(db, oid, pid, status, approved_by=""):
    _sql(db, "INSERT INTO outbox (id, prospect_id, to_email, status, approved_by) VALUES (?, ?, 'x@example.com', ?, ?)",
         (oid, pid, status, approved_by))


def test_reach_furthest_stage_and_reason():
    assert _funnel_reach({"email_status": "invalid"}) == ("found", "email invalid")
    assert _funnel_reach({"email_status": "risky"}) == ("found", "catch-all (not sent)")
    assert _funnel_reach({"email_status": "risky"}, send_to_risky=True) == ("deliverable", "not written yet")
    assert _funnel_reach({"status": "queued", "email_status": "verified", "drafted": 1, "pending": 1}) == \
        ("drafted", "waiting for your approval")
    assert _funnel_reach({"status": "queued", "drafted": 1, "approved": 1, "scheduled": 1}) == ("approved", "scheduled")
    # An Instantly send has no Outbox row but still counts as sent.
    assert _funnel_reach({"status": "contacted", "email_status": "verified"}) == ("sent", "no reply yet")
    assert _funnel_reach({"status": "contacted", "email_status": "invalid", "sent": 1}) == ("sent", "bounced")
    assert _funnel_reach({"status": "replied", "has_conversation": 1, "intent": "interested"})[0] == "interested"
    assert _funnel_reach({"status": "meeting"}) == ("meeting", "meeting")


def test_funnel_is_cumulative_and_filters(client):
    db = client.db_path
    asyncio.run(StateManager(str(db)).init_db())
    _sql(db, "INSERT INTO companies (id, name, industry, created_at) VALUES ('c1', 'Acme Ledger', 'accounting firm', "
             "'2026-01-01T10:00:00')")
    _sql(db, "INSERT INTO companies (id, name, industry, created_at) VALUES ('c2', 'Acme Studio', 'design studio', "
             "'2026-01-01T10:00:00')")
    _prospect(db, "p1", email_status="guess")
    _prospect(db, "p2", status="queued", email_status="verified")
    _outbox(db, "o2", "p2", "pending_review")
    _prospect(db, "p3", status="contacted", email_status="verified")
    _outbox(db, "o3", "p3", "sent", "you")
    _prospect(db, "p4", status="replied", email_status="verified", company="c2")
    _outbox(db, "o4", "p4", "sent", "you")
    _sql(db, "INSERT INTO conversations (id, prospect_id, intent, stage, created_at, updated_at) "
             "VALUES ('v4', 'p4', 'interested', 'qualifying', '2026-01-02', '2026-01-02')")

    data = client.get("/api/funnel").json()
    counts = {s["key"]: s["count"] for s in data["stages"]}
    assert counts == {"found": 4, "deliverable": 3, "drafted": 3, "approved": 2, "sent": 2,
                      "replied": 1, "interested": 1, "meeting": 0}
    held = {s["key"]: s["held"] for s in data["stages"]}
    assert held["found"] == {"email unverified": 1}
    assert held["drafted"] == {"waiting for your approval": 1}
    assert data["companies"] == 2 and data["companies_with_contact"] == 2
    assert {i["name"]: i["n"] for i in data["industries"]} == {"accounting firm": 3, "design studio": 1}

    one = client.get("/api/funnel?industry=design studio").json()
    assert [s["count"] for s in one["stages"]][:3] == [1, 1, 1] and one["companies"] == 1

    # Everything above was found long ago, so a 7-day window is empty.
    recent = client.get("/api/funnel?days=7").json()
    assert recent["stages"][0]["count"] == 0 and recent["companies"] == 0


def test_funnel_on_empty_install(client):
    data = client.get("/api/funnel?days=-5").json()
    assert data["days"] == 0 and all(s["count"] == 0 for s in data["stages"])
