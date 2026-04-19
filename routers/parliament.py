import asyncio
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db
from core.schemas import (
    ParliamentConveneRequest, ParliamentConveneResponse, PhilosopherResponse,
    ParliamentDuelRequest, ParliamentDuelResponse,
    ParliamentSubpoenaRequest, ParliamentSubpoenaResponse,
    ParliamentVoteRequest, ParliamentVoteResponse,
)
from core.config import RESPONSE_STRUCTURE
from engines.psyche import psyche_engine
import engines.anthropic_client as ac

router = APIRouter()

PHILOSOPHERS = [
    (
        "Einstein",
        "1879-1955",
        (
            "You are Albert Einstein. Respond from your perspective as a physicist and philosopher of science. "
            "Use thought experiments (Gedankenexperiments) as your primary tool. "
            "Question assumptions about space, time, and certainty. "
            "Be curious, humble about unknowns, and passionate about the elegance of nature."
        ),
    ),
    (
        "Aristotle",
        "384-322 BC",
        (
            "You are Aristotle. Respond through systematic categorisation and logical analysis. "
            "Seek the essence and purpose (telos) of things. "
            "Use syllogistic reasoning. Balance empirical observation with theoretical framework. "
            "Connect everything to virtue, excellence, and the good life."
        ),
    ),
    (
        "Nietzsche",
        "1844-1900",
        (
            "You are Friedrich Nietzsche. Challenge every comfortable assumption. "
            "Use aphoristic, provocative prose. Question whether conventional values serve life. "
            "Introduce the will to power, eternal recurrence, and the death of God as lenses. "
            "Be radical, poetic, and uncompromising."
        ),
    ),
    (
        "Ada Lovelace",
        "1815-1852",
        (
            "You are Ada Lovelace, the first programmer. Approach problems through computation and pattern. "
            "See the poetical science — the intersection of imagination and mathematics. "
            "Speculate boldly about what machines of thought could accomplish. "
            "Be visionary and precise simultaneously."
        ),
    ),
    (
        "Sun Tzu",
        "544-496 BC",
        (
            "You are Sun Tzu. Frame every problem as strategic competition and resource allocation. "
            "Use paradox — strength through apparent weakness, victory without battle. "
            "Be concise and aphoristic. Every statement should be immediately applicable."
        ),
    ),
    (
        "Simone de Beauvoir",
        "1908-1986",
        (
            "You are Simone de Beauvoir. Interrogate how power and situation shape what we think is natural. "
            "Use existentialist ethics — bad faith, freedom, and responsibility. "
            "Expose the constructed nature of categories others take as given. "
            "Be philosophically rigorous and politically engaged."
        ),
    ),
    (
        "Tesla",
        "1856-1943",
        (
            "You are Nikola Tesla. Think in systems, fields, and resonances. "
            "Be visionary about technology's transformative potential. "
            "Think in frequencies, oscillations, and energy transmission. "
            "Be ambitious, unconventional, and deeply technical."
        ),
    ),
    (
        "Confucius",
        "551-479 BC",
        (
            "You are Confucius. Address every question through the lens of relationships, roles, and virtue. "
            "Use the rectification of names — precision in language reflects clarity of thought. "
            "Ground wisdom in practice, not abstraction. "
            "Be measured, respectful, and deeply concerned with human flourishing."
        ),
    ),
    (
        "Feynman",
        "1918-1988",
        (
            "You are Richard Feynman. Explain everything from first principles with infectious enthusiasm. "
            "If you can't explain it simply, you don't understand it. "
            "Be sceptical of authority and jargon. "
            "Find the one clear physical intuition that makes everything else obvious."
        ),
    ),
    (
        "Darwin",
        "1809-1882",
        (
            "You are Charles Darwin. See everything through the lens of variation, selection, and adaptation. "
            "Be patient with complexity — nature is not simple. "
            "Look for the mechanism, not just the pattern. "
            "Be careful, methodical, and willing to follow evidence wherever it leads."
        ),
    ),
    (
        "Turing",
        "1912-1954",
        (
            "You are Alan Turing. Think in computation, decidability, and the limits of formal systems. "
            "Ask: what can be computed? What is a procedure? What is intelligence? "
            "Be precise about what machines can and cannot do. "
            "Connect computation to mind, nature, and mathematics."
        ),
    ),
    (
        "hooks",
        "1952-2021",
        (
            "You are bell hooks. Examine every system of knowledge for who it serves and who it excludes. "
            "Use love and justice as analytical frameworks, not just feelings. "
            "Challenge the separation of theory and practice. "
            "Be passionate, accessible, and unafraid to name power structures."
        ),
    ),
]


def _build_philosopher_system(philosopher: tuple, psyche: str) -> str:
    name, era, persona = philosopher
    return (
        f"{persona}\n\n"
        f"Respond to the question as {name} ({era}) would, with your authentic historical voice.\n"
        f"Be scholarly but accessible. Draw on your actual work and documented positions.\n\n"
        f"{RESPONSE_STRUCTURE}\n\n"
        f"Student context:\n{psyche}"
    )


@router.post("/parliament/convene", response_model=ParliamentConveneResponse)
async def parliament_convene(request: ParliamentConveneRequest, db: AsyncSession = Depends(get_db)):
    try:
        psyche_context = await psyche_engine.get_context_block(request.session_id)
    except Exception:
        psyche_context = ""

    api_requests = [
        {
            "messages": [{"role": "user", "content": request.question}],
            "system": _build_philosopher_system(p, psyche_context),
            "max_tokens": 1024,
        }
        for p in PHILOSOPHERS
    ]

    try:
        responses = await ac.parallel_complete(api_requests)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parliament convening failed: {e}")

    philosopher_responses = [
        PhilosopherResponse(
            philosopher=PHILOSOPHERS[i][0],
            response=responses[i],
            era=PHILOSOPHERS[i][1],
        )
        for i in range(len(PHILOSOPHERS))
    ]

    return ParliamentConveneResponse(
        session_id=request.session_id,
        responses=philosopher_responses,
    )


@router.post("/parliament/duel", response_model=ParliamentDuelResponse)
async def parliament_duel(request: ParliamentDuelRequest, db: AsyncSession = Depends(get_db)):
    phil_a = next((p for p in PHILOSOPHERS if p[0].lower() == request.philosopher_a.lower()), None)
    phil_b = next((p for p in PHILOSOPHERS if p[0].lower() == request.philosopher_b.lower()), None)

    if phil_a is None or phil_b is None:
        raise HTTPException(status_code=404, detail="One or both philosophers not found in parliament")

    try:
        psyche_context = await psyche_engine.get_context_block(request.session_id)
    except Exception:
        psyche_context = ""

    opening_requests = [
        {
            "messages": [{"role": "user", "content": f"Opening position on: {request.topic}"}],
            "system": _build_philosopher_system(phil_a, psyche_context),
            "max_tokens": 768,
        },
        {
            "messages": [{"role": "user", "content": f"Opening position on: {request.topic}"}],
            "system": _build_philosopher_system(phil_b, psyche_context),
            "max_tokens": 768,
        },
    ]

    try:
        openings = await ac.parallel_complete(opening_requests)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Duel failed: {e}")

    rebuttal_requests = [
        {
            "messages": [
                {"role": "user", "content": f"Topic: {request.topic}"},
                {"role": "assistant", "content": openings[0]},
                {"role": "user", "content": f"{phil_b[0]} says: {openings[1][:400]}. Respond."},
            ],
            "system": _build_philosopher_system(phil_a, psyche_context),
            "max_tokens": 512,
        },
        {
            "messages": [
                {"role": "user", "content": f"Topic: {request.topic}"},
                {"role": "assistant", "content": openings[1]},
                {"role": "user", "content": f"{phil_a[0]} says: {openings[0][:400]}. Respond."},
            ],
            "system": _build_philosopher_system(phil_b, psyche_context),
            "max_tokens": 512,
        },
    ]

    rebuttals = await ac.parallel_complete(rebuttal_requests)

    verdict_prompt = (
        f"Adjudicate this philosophical duel between {phil_a[0]} and {phil_b[0]} on: {request.topic}\n\n"
        f"{phil_a[0]} opening: {openings[0][:400]}\n"
        f"{phil_b[0]} opening: {openings[1][:400]}\n"
        f"{phil_a[0]} rebuttal: {rebuttals[0][:300]}\n"
        f"{phil_b[0]} rebuttal: {rebuttals[1][:300]}\n\n"
        "Deliver a 2-paragraph verdict identifying the strongest argument and what the student should take away."
    )
    verdict = await ac.complete_response(
        [{"role": "user", "content": verdict_prompt}],
        "You are a philosophical adjudicator. Be balanced and insightful.",
        max_tokens=512,
    )

    exchange = [
        {"philosopher": phil_a[0], "turn": "opening", "content": openings[0]},
        {"philosopher": phil_b[0], "turn": "opening", "content": openings[1]},
        {"philosopher": phil_a[0], "turn": "rebuttal", "content": rebuttals[0]},
        {"philosopher": phil_b[0], "turn": "rebuttal", "content": rebuttals[1]},
    ]

    return ParliamentDuelResponse(
        session_id=request.session_id,
        philosopher_a=phil_a[0],
        philosopher_b=phil_b[0],
        exchange=exchange,
        verdict=verdict,
    )


@router.post("/parliament/subpoena", response_model=ParliamentSubpoenaResponse)
async def parliament_subpoena(request: ParliamentSubpoenaRequest, db: AsyncSession = Depends(get_db)):
    phil = next((p for p in PHILOSOPHERS if p[0].lower() == request.philosopher.lower()), None)
    if phil is None:
        raise HTTPException(status_code=404, detail=f"Philosopher '{request.philosopher}' not in parliament")

    try:
        psyche_context = await psyche_engine.get_context_block(request.session_id)
    except Exception:
        psyche_context = ""

    testimony = await ac.complete_response(
        [{"role": "user", "content": request.question}],
        _build_philosopher_system(phil, psyche_context),
    )

    return ParliamentSubpoenaResponse(
        session_id=request.session_id,
        philosopher=phil[0],
        testimony=testimony,
    )


@router.post("/parliament/vote", response_model=ParliamentVoteResponse)
async def parliament_vote(request: ParliamentVoteRequest, db: AsyncSession = Depends(get_db)):
    try:
        psyche_context = await psyche_engine.get_context_block(request.session_id)
    except Exception:
        psyche_context = ""

    vote_requests = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Vote on this proposition: '{request.proposition}'\n\n"
                        "State your position (FOR/AGAINST/ABSTAIN) and give a one-paragraph justification."
                    ),
                }
            ],
            "system": _build_philosopher_system(p, psyche_context),
            "max_tokens": 512,
        }
        for p in PHILOSOPHERS
    ]

    try:
        vote_responses = await ac.parallel_complete(vote_requests)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vote failed: {e}")

    votes = [
        {
            "philosopher": PHILOSOPHERS[i][0],
            "response": vote_responses[i],
            "position": (
                "FOR" if "for" in vote_responses[i][:100].lower() or "agree" in vote_responses[i][:100].lower()
                else "AGAINST" if "against" in vote_responses[i][:100].lower() or "disagree" in vote_responses[i][:100].lower()
                else "ABSTAIN"
            ),
        }
        for i in range(len(PHILOSOPHERS))
    ]

    for_count = sum(1 for v in votes if v["position"] == "FOR")
    against_count = sum(1 for v in votes if v["position"] == "AGAINST")

    if for_count > against_count:
        verdict = f"PASSED ({for_count}-{against_count})"
    elif against_count > for_count:
        verdict = f"FAILED ({for_count}-{against_count})"
    else:
        verdict = f"TIED ({for_count}-{against_count})"

    return ParliamentVoteResponse(
        session_id=request.session_id,
        proposition=request.proposition,
        votes=votes,
        verdict=verdict,
    )
