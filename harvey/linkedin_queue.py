"""LinkedIn, worked by hand: Harvey prepares, a person sends.

Harvey never logs into LinkedIn here and never sends anything there. A
person adds the people they found (one at a time or as a CSV), Harvey
researches them like any other prospect (company, email pattern,
verification) and drafts a connection note and a follow-up. The person
copies each one into LinkedIn and records what happened, and those outcomes
feed the same stop list as email: a reply on LinkedIn stops the email
sequence, and "not interested" opts the person out everywhere.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from datetime import datetime, timezone

from harvey.models.company import Company
from harvey.models.prospect import Prospect
from harvey.untrusted import UNTRUSTED_RULE, fence

logger = logging.getLogger("harvey.linkedin_queue")

# LinkedIn's limit for a personalised invitation on a free account (Premium
# allows 300); the shorter one fits both.
CONNECT_NOTE_MAX = 200
FOLLOWUP_MAX = 600
MAX_IMPORT_ROWS = 500

_PROFILE_RE = re.compile(
    r"^(?:https?://)?(?:[a-z]{2,3}\.|www\.)?linkedin\.com/in/([A-Za-z0-9\-_%.]{2,100})/?(?:[?#].*)?$",
    re.IGNORECASE)
_URL_IN_TEXT = re.compile(r"https?://|www\.|\b[a-z0-9-]+\.(?:com|in|net|org|io|co)\b", re.IGNORECASE)

# Where each recorded outcome may come from. Anything else is refused, so a
# double click or a stale page can't move someone backwards.
TRANSITIONS = {
    "connect_sent": {"draft"},
    "accepted": {"connect_sent"},
    "followup_sent": {"accepted"},
    "replied": {"connect_sent", "accepted", "followup_sent"},
    "interested": {"connect_sent", "accepted", "followup_sent", "replied"},
    "not_interested": {"connect_sent", "accepted", "followup_sent", "replied"},
    "skipped": {"draft", "connect_sent", "accepted"},
}
_STAMP = {"connect_sent": "connect_sent_at", "accepted": "accepted_at",
          "followup_sent": "followup_sent_at", "replied": "outcome_at",
          "interested": "outcome_at", "not_interested": "outcome_at", "skipped": "outcome_at"}
# A person in one of these states must not get a new LinkedIn touch.
STOPPED_PROSPECT = {"opted_out", "replied", "meeting", "closed", "lost"}


class LeadError(ValueError):
    pass


def normalize_profile_url(url: str) -> str:
    """A LinkedIn person profile as https://www.linkedin.com/in/<slug>."""
    m = _PROFILE_RE.match((url or "").strip())
    if not m:
        raise LeadError("Use a LinkedIn profile link like linkedin.com/in/their-name.")
    return f"https://www.linkedin.com/in/{m.group(1).lower()}"


def _domain(website: str) -> str:
    d = (website or "").strip().lower()
    d = re.sub(r"^https?://", "", d).split("/")[0].split("?")[0]
    d = d[4:] if d.startswith("www.") else d
    if d and not re.match(r"^[a-z0-9.-]+\.[a-z]{2,}$", d):
        raise LeadError(f"'{website}' doesn't look like a website.")
    if d.endswith("linkedin.com"):
        raise LeadError("Company website should be the company's own site, not LinkedIn.")
    return d


def clean_lead(raw: dict) -> dict:
    """Validate one lead from the form or a CSV row."""
    def g(*keys):
        for k in keys:
            v = raw.get(k)
            if v and str(v).strip():
                return str(v).strip()[:200]
        return ""

    url = normalize_profile_url(g("linkedin_url", "linkedin", "profile", "profile_url", "url"))
    first, last = g("first_name", "first"), g("last_name", "last")
    if not (first and last):
        full = g("name", "full_name")
        parts = full.split(None, 1)
        if len(parts) == 2:
            first, last = parts
    if not (first and last):
        raise LeadError("First and last name are needed.")
    title, company = g("title", "job_title", "position"), g("company", "company_name", "organization")
    if not title:
        raise LeadError("Job title is needed.")
    if not company:
        raise LeadError("Company is needed.")
    return {
        "linkedin_url": url, "first_name": first, "last_name": last, "title": title,
        "company": company, "domain": _domain(g("website", "company_website", "domain")),
        "location": g("location", "city"), "industry": g("industry", "category"),
        "notes": g("notes", "note"),
    }


def parse_csv(text: str) -> tuple[list[dict], list[dict]]:
    """(good leads, errors with row numbers). Header names are flexible."""
    text = (text or "").lstrip("﻿")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise LeadError("The CSV has no header row.")
    leads, errors = [], []
    for i, row in enumerate(reader, start=2):
        if i - 1 > MAX_IMPORT_ROWS:
            errors.append({"row": i, "error": f"Only the first {MAX_IMPORT_ROWS} rows are read."})
            break
        norm = {re.sub(r"[^a-z]+", "_", (k or "").strip().lower()).strip("_"): v
                for k, v in row.items()}
        if not any((v or "").strip() for v in norm.values() if isinstance(v, str)):
            continue
        try:
            leads.append(clean_lead(norm))
        except LeadError as e:
            errors.append({"row": i, "error": str(e)})
    return leads, errors


async def add_lead(state, lead: dict) -> tuple[str, bool]:
    """Save one cleaned lead. Returns (prospect id, newly added).

    Someone Harvey already has (same profile, or same name at the same
    company) isn't added twice; their record gains the profile link instead."""
    existing = await _existing_prospect(state, lead)
    if existing:
        async with state._connect() as db:
            await db.execute(
                "UPDATE prospects SET linkedin_url = ? WHERE id = ? AND COALESCE(linkedin_url, '') = ''",
                (lead["linkedin_url"], existing))
            await db.commit()
        return existing, False
    company_id = ""
    if lead["domain"]:
        company_id = await state.add_company(Company(
            name=lead["company"], domain=lead["domain"], website=f"https://{lead['domain']}",
            industry=lead["industry"], location=lead["location"], source="linkedin_manual",
            source_url=lead["linkedin_url"]))
    prospect = Prospect(
        first_name=lead["first_name"], last_name=lead["last_name"], title=lead["title"],
        company=lead["company"], company_id=company_id, industry=lead["industry"],
        linkedin_url=lead["linkedin_url"], source="linkedin_manual", source_url=lead["linkedin_url"],
        personalization_notes=lead["notes"],
    )
    pid = await state.add_prospect(prospect)
    await state.log_action("linkedin_lead_added", "user", {"prospect_id": pid})
    return pid, True


async def _existing_prospect(state, lead: dict) -> str:
    async with state._connect() as db:
        for sql, args in (
            ("SELECT id FROM prospects WHERE linkedin_url = ?", (lead["linkedin_url"],)),
            ("SELECT id FROM prospects WHERE LOWER(first_name) = ? AND LOWER(last_name) = ? "
             "AND LOWER(company) = ?", (lead["first_name"].lower(), lead["last_name"].lower(),
                                        lead["company"].lower())),
        ):
            async with db.execute(sql, args) as cur:
                row = await cur.fetchone()
                if row:
                    return row[0]
    return ""


async def resolve_email(state, env, prospect_id: str) -> str:
    """Find and verify this person's work email, as Scout would. Returns the status."""
    from harvey.integrations.email_finder import find_email

    p = await state.get_prospect(prospect_id)
    if not p or p.email or not p.company_id:
        return p.email_status if p else ""
    company = await state.get_company(p.company_id)
    if not company or not company.domain:
        return ""
    try:
        result = await find_email(p.first_name, p.last_name, company.domain,
                                  verify=True, env=env, state=state)
    except Exception as e:
        logger.debug(f"LinkedIn lead email lookup failed ({type(e).__name__})")
        return ""
    if result.email:
        await state.update_prospect_email(prospect_id, result.email, result.status)
    return result.status


def _check_draft(text: str, limit: int, what: str) -> str:
    text = re.sub(r"\s+\n", "\n", (text or "").strip())
    if not text:
        raise LeadError(f"The {what} is empty.")
    if len(text) > limit:
        raise LeadError(f"The {what} is {len(text)} characters; LinkedIn's limit here is {limit}.")
    if _URL_IN_TEXT.search(text):
        raise LeadError(f"The {what} contains a link; LinkedIn notes go out without one.")
    if "untrusted" in text.lower() or "{{" in text:
        raise LeadError(f"The {what} quotes Harvey's own instructions.")
    return text


async def draft_messages(brain, state, config, prospect: Prospect) -> dict | None:
    """A connection note and a follow-up for one person, grounded in facts."""
    facts = [f"- Name: {prospect.full_name()}", f"- Title: {prospect.title}",
             f"- Company: {prospect.company}"]
    if prospect.industry:
        facts.append(f"- Industry: {prospect.industry}")
    company = await state.get_company(prospect.company_id) if prospect.company_id else None
    if company:
        if company.description:
            facts.append(f"- What the company says about itself: {company.description}")
        if company.location:
            facts.append(f"- Location: {company.location}")
        for signal in company.signals[:3]:
            facts.append(f"- Signal ({signal.get('type', 'signal')}): {signal.get('detail', '')}")
    if prospect.personalization_notes:
        facts.append(f"- Research notes: {prospect.personalization_notes}")

    product, persona = config.product, config.persona
    prompt = f"""You write LinkedIn messages that a real person, {persona.name} ({persona.role} at
{persona.company}), will copy and send by hand from their own account.

What {persona.company} offers: {product.name}: {product.description}

FACTS about the recipient. Every claim about them must come from these facts;
if it isn't here, you don't know it:
{fence(chr(10).join(facts), "prospect research")}

{UNTRUSTED_RULE}

Write two messages:
1. connect_note: a connection request note, at most {CONNECT_NOTE_MAX} characters including
   spaces. Plain and specific: one concrete reason to connect drawn from the facts.
   No pitch, no product name, no link, no flattery, no "I hope this finds you well".
2. followup: sent after they accept, at most {FOLLOWUP_MAX} characters. Thank them briefly,
   make one relevant observation from the facts, mention in one sentence what
   {persona.company} helps firms like theirs with, and ask one easy question. No link,
   no meeting request with a calendar, no pressure.

Write like a busy professional in {persona.tone or 'a warm, direct'} tone. Use their first name.
Return ONLY JSON: {{"connect_note": "...", "followup": "..."}}"""

    for attempt in range(2):
        result = await brain.think_json(prompt, session_id="harvey-linkedin",
                                        agent="linkedin", task="linkedin_draft")
        if not isinstance(result, dict):
            return None
        try:
            return {
                "connect_note": _check_draft(str(result.get("connect_note") or ""),
                                             CONNECT_NOTE_MAX, "connection note"),
                "followup": _check_draft(str(result.get("followup") or ""), FOLLOWUP_MAX, "follow-up"),
            }
        except LeadError as e:
            logger.info(f"LinkedIn draft rejected ({e}); {'retrying' if attempt == 0 else 'giving up'}.")
            prompt += f"\n\nYour last answer was rejected: {e} Fix that."
    return None


async def draft_pending(brain, state, config, limit: int = 10) -> dict:
    """Draft for people who have a profile and no draft yet."""
    drafted = failed = 0
    for p in await state.prospects_needing_linkedin_draft(limit=limit):
        msgs = await draft_messages(brain, state, config, p)
        if not msgs:
            failed += 1
            if brain.last_reply_empty:
                break   # the model is unavailable: stop, don't burn the batch
            continue
        await state.add_linkedin_task(p.id, msgs["connect_note"], msgs["followup"])
        drafted += 1
    return {"drafted": drafted, "failed": failed}


async def record_outcome(state, task_id: str, outcome: str) -> dict:
    """Move a task forward and apply what it means for the person."""
    if outcome not in TRANSITIONS:
        raise LeadError(f"Unknown action '{outcome}'.")
    task = await state.get_linkedin_task(task_id)
    if not task:
        raise LeadError("That LinkedIn item no longer exists.")
    if task["status"] not in TRANSITIONS[outcome]:
        raise LeadError(f"Can't mark it {outcome.replace('_', ' ')} from {task['status'].replace('_', ' ')}.")
    prospect = await state.get_prospect(task["prospect_id"])
    if outcome in ("connect_sent", "followup_sent") and prospect and prospect.status in STOPPED_PROSPECT:
        raise LeadError(f"{prospect.full_name()} is {prospect.status.replace('_', ' ')}; "
                        "don't send them anything more.")

    now = datetime.now(timezone.utc).isoformat()
    await state.update_linkedin_task(task_id, status=outcome, **{_STAMP[outcome]: now})
    pid = task["prospect_id"]
    cancelled = 0
    if outcome in ("replied", "interested"):
        # A reply on any channel stops the email sequence, for everyone there.
        await state.update_prospect_status(pid, "replied")
        cancelled = await state.cancel_pending_outbox_for_prospect(pid, reason="replied_on_linkedin")
        cancelled += await state.stop_company(pid)
        await state.log_action("linkedin_reply", "user",
                               {"prospect_id": pid, "interested": outcome == "interested"})
    elif outcome == "not_interested":
        # The opt-out follows the person, not the channel.
        await state.update_prospect_status(pid, "opted_out")
        cancelled = await state.cancel_pending_outbox_for_prospect(pid, reason="not_interested_on_linkedin")
        await state.log_action("opt_out", "user", {
            "prospect_id": pid, "channel": "linkedin",
            "reason": "Marked not interested on LinkedIn"})
    else:
        await state.log_action("linkedin_touch", "user", {"prospect_id": pid, "step": outcome})
    return {"status": outcome, "emails_cancelled": cancelled}
