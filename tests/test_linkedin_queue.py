"""The LinkedIn queue: add finds by hand, Harvey drafts, a person sends."""

import asyncio
import sqlite3

import pytest

from harvey import linkedin_queue as lq
from harvey.models.prospect import Prospect
from harvey.state import StateManager
from tests.test_dashboard_signals import client  # noqa: F401  (fixture)

LEAD = {"linkedin_url": "https://in.linkedin.com/in/Pat-Example-123?trk=x", "first_name": "Pat",
        "last_name": "Example", "title": "Managing Partner", "company": "Example Ledger LLP",
        "website": "https://www.example.com/about"}


@pytest.fixture
def state(tmp_path):
    s = StateManager(str(tmp_path / "harvey.db"))
    asyncio.run(s.init_db())
    return s


def test_profile_urls():
    assert lq.normalize_profile_url("linkedin.com/in/pat-example/") == "https://www.linkedin.com/in/pat-example"
    assert lq.normalize_profile_url("https://in.linkedin.com/in/Pat?x=1") == "https://www.linkedin.com/in/pat"
    for bad in ("https://www.linkedin.com/company/example", "https://example.com/in/pat",
                "javascript:alert(1)//linkedin.com/in/pat", ""):
        with pytest.raises(lq.LeadError):
            lq.normalize_profile_url(bad)


def test_clean_lead_and_csv():
    lead = lq.clean_lead(LEAD)
    assert lead["linkedin_url"] == "https://www.linkedin.com/in/pat-example-123"
    assert lead["domain"] == "example.com"
    with pytest.raises(lq.LeadError):
        lq.clean_lead({**LEAD, "title": ""})
    with pytest.raises(lq.LeadError):
        lq.clean_lead({**LEAD, "website": "https://linkedin.com/company/x"})
    csv_text = ("﻿LinkedIn URL,Name,Title,Company,Website\n"
                "linkedin.com/in/sam-sample,Sam Sample,Partner,Sample & Co,sample.example\n"
                "linkedin.com/company/nope,Kim Test,Owner,Test Co,\n"
                ",,,,\n")
    leads, errors = lq.parse_csv(csv_text)
    assert [l["first_name"] for l in leads] == ["Sam"] and leads[0]["last_name"] == "Sample"
    assert errors == [{"row": 3, "error": "Use a LinkedIn profile link like linkedin.com/in/their-name."}]


def test_add_lead_dedups_and_links_existing(state):
    pid, new = asyncio.run(lq.add_lead(state, lq.clean_lead(LEAD)))
    assert new and pid
    again, new2 = asyncio.run(lq.add_lead(state, lq.clean_lead(LEAD)))
    assert again == pid and not new2
    # Someone Scout already found gains the profile link instead of a duplicate.
    other = asyncio.run(state.add_prospect(Prospect(first_name="Alex", last_name="Placeholder",
                                                   title="Partner", company="Acme Audit")))
    lead = lq.clean_lead({**LEAD, "linkedin_url": "linkedin.com/in/alex-p", "first_name": "Alex",
                          "last_name": "Placeholder", "company": "Acme Audit"})
    got, new3 = asyncio.run(lq.add_lead(state, lead))
    assert got == other and not new3
    assert asyncio.run(state.get_prospect(other)).linkedin_url == "https://www.linkedin.com/in/alex-p"


class FakeBrain:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.last_reply_empty = False

    async def think_json(self, prompt, **kw):
        self.prompts.append(prompt)
        r = self.replies.pop(0) if self.replies else None
        self.last_reply_empty = r is None
        return r


class Cfg:
    class product:
        name, description = "Example Vault", "secure client document portal"

    class persona:
        name, role, company, tone = "Jo Sender", "Founder", "Example Labs", "warm, direct"


def test_drafts_are_checked_and_retried(state):
    pid, _ = asyncio.run(lq.add_lead(state, lq.clean_lead({**LEAD, "notes": "Posted about audit season"})))
    too_long = {"connect_note": "x" * 201, "followup": "ok"}
    with_link = {"connect_note": "Hi Pat, see https://example.com", "followup": "ok"}
    good = {"connect_note": "Hi Pat, I work with CA firms in Pune on client document handling. Happy to connect.",
            "followup": "Thanks for connecting, Pat. Quick question: how do clients send you documents today?"}
    brain = FakeBrain([too_long, good])
    out = asyncio.run(lq.draft_pending(brain, state, Cfg))
    assert out == {"drafted": 1, "failed": 0}
    assert "rejected" in brain.prompts[1] and "Posted about audit season" in brain.prompts[0]
    assert "<untrusted" in brain.prompts[0]
    item = asyncio.run(state.linkedin_queue())[0]
    assert item["prospect_id"] == pid and item["status"] == "draft"
    assert item["connect_note"] == good["connect_note"]
    # Already drafted: nothing more to do.
    assert asyncio.run(lq.draft_pending(FakeBrain([with_link]), state, Cfg)) == {"drafted": 0, "failed": 0}


def test_unavailable_model_stops_the_batch(state):
    for i in range(3):
        asyncio.run(lq.add_lead(state, lq.clean_lead({**LEAD, "linkedin_url": f"linkedin.com/in/p{i}",
                                                     "first_name": f"P{i}"})))
    brain = FakeBrain([])
    assert asyncio.run(lq.draft_pending(brain, state, Cfg)) == {"drafted": 0, "failed": 1}


def _task(state, status_prospect="new"):
    pid, _ = asyncio.run(lq.add_lead(state, lq.clean_lead(LEAD)))
    tid = asyncio.run(state.add_linkedin_task(pid, "Hi Pat", "Thanks Pat"))
    if status_prospect != "new":
        asyncio.run(state.update_prospect_status(pid, status_prospect))
    return pid, tid


def test_outcomes_move_forward_only(state):
    pid, tid = _task(state)
    with pytest.raises(lq.LeadError):
        asyncio.run(lq.record_outcome(state, tid, "accepted"))      # not sent yet
    asyncio.run(lq.record_outcome(state, tid, "connect_sent"))
    with pytest.raises(lq.LeadError):
        asyncio.run(lq.record_outcome(state, tid, "connect_sent"))  # double click
    asyncio.run(lq.record_outcome(state, tid, "accepted"))
    asyncio.run(lq.record_outcome(state, tid, "followup_sent"))
    task = asyncio.run(state.get_linkedin_task(tid))
    assert task["status"] == "followup_sent" and task["connect_sent_at"] and task["followup_sent_at"]


def test_reply_on_linkedin_stops_email(state):
    pid, tid = _task(state)
    with sqlite3.connect(state.db_path) as db:
        db.execute("INSERT INTO outbox (id, prospect_id, to_email, status, kind) "
                   "VALUES ('o1', ?, 'pat@example.com', 'pending_review', 'sequence')", (pid,))
    asyncio.run(lq.record_outcome(state, tid, "connect_sent"))
    out = asyncio.run(lq.record_outcome(state, tid, "replied"))
    assert out["emails_cancelled"] >= 1
    assert asyncio.run(state.get_prospect(pid)).status == "replied"
    with sqlite3.connect(state.db_path) as db:
        assert db.execute("SELECT status FROM outbox WHERE id = 'o1'").fetchone()[0] == "cancelled"


def test_not_interested_opts_out_everywhere(state):
    pid, tid = _task(state)
    asyncio.run(lq.record_outcome(state, tid, "connect_sent"))
    asyncio.run(lq.record_outcome(state, tid, "not_interested"))
    assert asyncio.run(state.get_prospect(pid)).status == "opted_out"
    assert asyncio.run(state.opt_out_history(pid))


def test_stopped_person_gets_no_new_touch(state):
    pid, tid = _task(state, status_prospect="opted_out")
    with pytest.raises(lq.LeadError):
        asyncio.run(lq.record_outcome(state, tid, "connect_sent"))
    asyncio.run(lq.record_outcome(state, tid, "skipped"))
    assert asyncio.run(state.prospects_needing_linkedin_draft()) == []


def test_dashboard_flow(client, monkeypatch):
    from harvey import dashboard as dash
    monkeypatch.setattr(dash, "_start_linkedin_worker", lambda draft=True: None)
    r = client.post("/api/linkedin/leads", json={"lead": LEAD}).json()
    assert r["success"] and r["added"] == 1
    assert client.post("/api/linkedin/leads", json={"lead": {**LEAD, "title": ""}}).status_code == 400
    r = client.post("/api/linkedin/leads", json={"csv": "linkedin_url,name,title,company\n"
                                                         "linkedin.com/in/sam-s,Sam Sample,Owner,Sample Co\n"}).json()
    assert r["added"] == 1 and r["error_count"] == 0
    q = client.get("/api/linkedin/queue").json()
    assert q["waiting_for_draft"] == 2 and q["items"] == [] and q["daily_cap"] >= 1

    state = StateManager(str(client.db_path))
    pid = asyncio.run(state.prospects_needing_linkedin_draft())[0].id
    tid = asyncio.run(state.add_linkedin_task(pid, "Hi there", "Thanks"))
    assert client.post(f"/api/linkedin/{tid}/edit", json={"connect_note": "x" * 300}).status_code == 400
    assert client.post(f"/api/linkedin/{tid}/edit", json={"connect_note": "Hi, glad to connect."}).json()["success"]
    assert client.post(f"/api/linkedin/{tid}/mark", json={"outcome": "connect_sent"}).json()["success"]
    assert client.post(f"/api/linkedin/{tid}/mark", json={"outcome": "connect_sent"}).status_code == 409
    q = client.get("/api/linkedin/queue").json()
    assert q["sent_last_24h"] == 1 and q["items"][0]["status"] == "connect_sent"

    funnel = {s["key"]: s["count"] for s in client.get("/api/funnel").json()["stages"]}
    assert funnel["sent"] == 1
