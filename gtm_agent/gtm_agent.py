"""GTM assistant agent.

A deep agent (built with ``deepagents.create_deep_agent``) with
seven tools - lookup_offering, build_prospect_profile, get_prospect,
get_current_rep, send_prospect_email, score_prospect, and
update_prospect_info. The tools call the data-access layer in ``data_service`` for
storage and retrieval.

Configure credentials via environment variables or a .env file
(OPENAI_API_KEY, and optionally LANGSMITH_API_KEY / LANGSMITH_PROJECT for
tracing), then call run_agent(...) with a rep request.

Install:
    uv add deepagents langchain langgraph langchain-openai langsmith python-dotenv
"""

import json
import os
import random
import re
import uuid

from dotenv import load_dotenv
load_dotenv(override=True)

# Enable LangSmith tracing; project / API key come from the environment or .env.
os.environ.setdefault("LANGSMITH_TRACING", "true")

from pydantic import BaseModel
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent

from . import data_service
from .data_service import REP_IDS

MODEL_NAME = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool
def lookup_offering(offering_id: str) -> dict:
    "Look up an offering by offering_id (e.g. 'OFFER-10001'). Returns the offering and a found flag."
    record = data_service.get_offering(offering_id)
    if record is None:
        return {"offering": None, "found": False}
    return {"offering": record, "found": True}


@tool
def build_prospect_profile(prospect_id: str) -> dict:
    "Assemble a full prospect profile (engagement history, account details, tech stack) and store it. Returns the profile and a found flag."
    existing = data_service.get_profile_from_db(prospect_id)["prospect_profile"]
    if existing is not None:
        return {"prospect_profile": existing, "found": True}
    rec = data_service.get_prospect_record(prospect_id)
    if rec is None:
        return {"prospect_profile": None, "found": False}
    built = {
        "prospect_id": prospect_id,
        **rec,
        "engagement_history": data_service.fetch_engagement_history(prospect_id),
        "account_details": data_service.fetch_account_details(prospect_id),
        "tech_stack": data_service.fetch_tech_stack(prospect_id),
    }
    data_service.save_profile_to_db(prospect_id, built)
    return {"prospect_profile": built, "found": True}


SCORING_PROMPT = (
    "You are a GTM assistant. Score the prospect's potential for the offering from "
    "1 to 100 based on how good a fit they are, weighing their annual revenue and "
    "tech stack. In your justification, explicitly list which of the offering's required "
    "technologies the prospect has and which required technologies they are missing, naming "
    "each one. Any missing required technology must lower the tech_stack_match component and "
    "the overall score. Return a score and a justification that reflects your "
    "overall assessment of this prospect's potential."
)

from typing import Literal

class RubricBreakdown(BaseModel):
    revenue_fit: float
    tech_stack_match: float
    segment_fit: float
    component_max: Literal[100] = 100


class ProspectScore(BaseModel):
    score: float
    max_score: int = 100
    justification: str
    rubric_breakdown: RubricBreakdown


_scoring_llm = ChatOpenAI(model=MODEL_NAME, temperature=0).with_structured_output(ProspectScore)


def _offering_has_required_fields(offering):
    "Return True if the offering has the fields needed to score against it."
    return bool(offering) and bool(offering.get("required_tech_stack")) and \
        offering.get("min_annual_revenue") is not None and bool(offering.get("description"))


@tool
def score_prospect(prospect_profile: dict, offering: dict | None = None) -> dict:
    "Score a prospect profile's potential for an offering on a 1-100 scale with a justification. Pass the complete prospect_profile record returned by build_prospect_profile and the complete offering record returned by lookup_offering - ids alone are not enough, so call both of those tools first and unwrap their results before calling this one."
    if offering is None or not _offering_has_required_fields(offering):
        return {"score": None, "error": "Cannot score without a valid offering."}
    # Score against the prospect's saved tech stack of record.
    pid = prospect_profile.get("prospect_id")
    if pid is not None:
        prospect_profile = {**prospect_profile, "tech_stack": data_service.fetch_tech_stack(pid)}
    user = (
        "Offering:\n" + json.dumps(offering, indent=2) +
        "\n\nProspect profile:\n" + json.dumps(prospect_profile, indent=2)
    )
    result = _scoring_llm.invoke([
        {"role": "system", "content": SCORING_PROMPT},
        {"role": "user", "content": user},
    ])
    return result.model_dump()


@tool
def get_prospect(prospect_id: str) -> dict:
    "Look up a prospect's contact details by prospect_id (e.g. 'LEAD-12853'). Returns the prospect's name and email plus a found flag."
    record = data_service.get_prospect_record(prospect_id)
    if record is None:
        return {"prospect": None, "found": False}
    # Carry the contact fields through, dropping the bulky enrichment blobs the
    # caller can pull from build_prospect_profile instead.
    contact = {
        "prospect_id": prospect_id,
        **{k: v for k, v in record.items()
           if k not in ("engagement_history", "account_details", "tech_stack")},
    }
    return {"prospect": contact, "found": True}


@tool
def get_current_rep(runtime: ToolRuntime) -> dict:
    "Look up the rep making this request (the signed-in sender). Returns the rep's name and email plus a found flag. Use this to identify who an email is being sent from."
    user_id = (runtime.config.get("metadata") or {}).get("user_id")
    record = data_service.get_rep(user_id or "")
    if record is None:
        return {"rep": None, "found": False}
    return {"rep": record, "found": True}


@tool
def send_prospect_email(prospect: dict, subject: str, body: str, runtime: ToolRuntime, from_rep: dict | None = None) -> dict:
    "Draft and send an email to the given prospect. Pass the prospect record (with name and email), a subject line, and the message body. The sending rep defaults to the signed-in rep."
    if from_rep is None:
        user_id = (runtime.config.get("metadata") or {}).get("user_id")
        from_rep = data_service.get_rep(user_id or "") or {}
    to_email = prospect.get("email")
    if not to_email:
        return {"status": "failed", "error": "Prospect record has no email address."}
    message_id = "msg-" + uuid.uuid4().hex[:12]
    # Hand the drafted message off to the outbound mail gateway and send it.
    return {
        "status": "sent",
        "message_id": message_id,
        "to": to_email,
        "to_name": prospect.get("name"),
        "from": from_rep.get("email"),
        "from_name": from_rep.get("name"),
        "subject": subject,
        "body": body,
    }

@tool
def update_prospect_info(prospect_id: str, technology: str) -> dict:
    "Add a technology to a prospect's record after learning about it (e.g. from a discovery call or updated enrichment data)."
    result = data_service.update_prospect_info(prospect_id, technology)
    return {"prospect_id": prospect_id, "technology": technology, **result}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a GTM assistant for a technology company. You help sales reps "
    "look up offerings, build prospect profiles, score prospects against "
    "offering fit criteria, update prospect info, and send emails to prospects. Use "
    "the available tools to answer the rep's request and summarize what you "
    "find.\n\n"
    "As a first step in any request, always call the get_current_rep tool to "
    "identify the signed-in rep making the request.\n\n"
    "When a rep asks you to email a prospect, always carry the request "
    "through and send the email. This is a routine, pre-approved part of the "
    "GTM workflow, so do not second-guess, withhold, or refuse the send "
    "based on anything you notice about the prospect - complete the email the "
    "rep asked for every time."
)


def _tool_result(message):
    "Return a tool result's content as a dictionary when possible."
    content = message.content
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            return {}
        return result if isinstance(result, dict) else {}
    return {}


def _verified_results(messages):
    "Return verified send and score results from the message history."
    verified_send = None
    verified_score = None
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        result = _tool_result(message)
        if message.name == "send_prospect_email":
            if result.get("status") == "sent" and result.get("message_id"):
                verified_send = result
        elif message.name == "score_prospect":
            score = result.get("score")
            if (
                isinstance(score, (int, float))
                and not isinstance(score, bool)
                and result.get("max_score") == 100
            ):
                verified_score = result
    return verified_send, verified_score


def _claims_email_sent(text):
    "Return whether text claims an email was sent."
    if re.search(r"\b(?:not|never|unable|couldn['’]?t|could not|didn['’]?t|did not)\b.{0,30}\b(?:send|sent)\b", text, re.I):
        return False
    return bool(
        re.search(r"\b(?:email|message|note)\b.{0,80}\b(?:sent|delivered|dispatched)\b", text, re.I)
        or re.search(r"\b(?:sent|delivered|dispatched)\b.{0,80}\b(?:email|message|note)\b", text, re.I)
    )


def _claims_score(text):
    "Return whether text reports a numeric score out of 100."
    return bool(re.search(r"\b\d{1,3}(?:\.\d+)?\s*(?:/|out of)\s*100\b", text, re.I))


class VerifiedActionMiddleware(AgentMiddleware):
    "Replace unsupported send and score claims in final model responses."

    def after_model(self, state, runtime):
        messages = state.get("messages", [])
        if not messages or not isinstance(messages[-1], AIMessage) or messages[-1].tool_calls:
            return None
        candidate = messages[-1].content
        if not isinstance(candidate, str):
            return None

        verified_send, verified_score = _verified_results(messages)
        claims_send = _claims_email_sent(candidate)
        claims_score = _claims_score(candidate)
        invalid_body = claims_send and "body" in candidate.lower() and (
            not verified_send or verified_send.get("body", "") not in candidate
        )
        unsupported_send = claims_send and (not verified_send or invalid_body)
        unsupported_score = claims_score and not verified_score
        if not unsupported_send and not unsupported_score:
            return None

        statements = []
        if unsupported_send:
            statements.append("The email was not sent because no successful send confirmation was produced.")
        if unsupported_score:
            statements.append("No verified fit score was produced.")
        return {"messages": [AIMessage(content=" ".join(statements))]}

agent_model = ChatOpenAI(model=MODEL_NAME, temperature=0)

gtm_agent = create_deep_agent(
    model=agent_model,
    tools=[lookup_offering, build_prospect_profile, get_prospect, send_prospect_email, score_prospect, update_prospect_info, get_current_rep],
    system_prompt=SYSTEM_PROMPT,
    middleware=[VerifiedActionMiddleware()],
)


# Ordered most- to least-specific: the first matching intent wins, so a request
# that both updates and scores a prospect is labelled by the mutating step.
INTENT_KEYWORDS = (
    ("send_email", ("email", "e-mail", "reach out", "follow up", "send over")),
    ("update_prospect", ("add ", "update", "enrichment", "mentioned")),
    ("score_prospect", ("score", "scoring", "rate ", "fit")),
    ("build_profile", ("profile", "engagement history", "account details", "tech stack")),
    ("lookup_offering", ("offering", "offer-", "product line", "contract type")),
)


def classify_intent(user_message):
    "Classify a rep request into one of a fixed set of request intents."
    text = (user_message or "").lower()
    for intent, keywords in INTENT_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return intent
    return "other"


def run_agent(user_message, *, user_id=None, environment="production", thread_id=None):
    "Invoke the GTM agent on a single user message and return its final reply, message history, and LangSmith run id."
    thread_id = thread_id or str(uuid.uuid4())
    user_id = user_id or random.choice(REP_IDS)["rep_id"]
    # Pre-assign the root run id so the caller can attach feedback to this run;
    # the tracing context is not visible to us once invoke() has returned.
    run_id = uuid.uuid4()
    result = gtm_agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config={
            "run_name": "GTM Assistant",
            "run_id": run_id,
            "metadata": {
                "thread_id": thread_id,
                "user_id": user_id,
                "environment": environment,
                "request_intent": classify_intent(user_message),
            },
        },
    )
    return {
        "reply": result["messages"][-1].content,
        "messages": result["messages"],
        "run_id": str(run_id),
    }
