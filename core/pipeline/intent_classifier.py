"""
Fast (<1ms) regex-based intent classifier.
No LLM call — pure pattern matching on the user message.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field


@dataclass
class RouterDecision:
    intent: str                     # coding | reasoning | math | creative | vision | fast | default
    scores: dict[str, int] = field(default_factory=dict)
    has_image: bool = False
    is_long_doc: bool = False        # message > 4000 chars
    estimated_tokens: int = 0


_CODING = [
    r'\b(code|function|class|debug|fix|implement|algorithm|script|program|refactor)\b',
    r'\b(python|javascript|typescript|rust|go|java|sql|bash|c\+\+|kotlin|swift)\b',
    r'```',
    r'\b(def |class |import |const |async def|fn |func |public static)\b',
    r'\b(bug|error|exception|traceback|stacktrace|compile|runtime)\b',
    r'\b(api|endpoint|database|query|schema|migration|docker|kubernetes)\b',
    r'\b(test|unittest|pytest|jest|spec|mock|assert)\b',
    r'\b(array|list|dict|object|loop|recursion|sorting|tree|graph|hash)\b',
]

_REASONING = [
    r'\b(why|explain|analyze|compare|evaluate|argue|discuss|assess)\b',
    r'\b(think through|reason|consider|implications|tradeoffs|pros and cons)\b',
    r'\b(philosophy|ethics|consciousness|meaning|truth|justice|morality)\b',
    r'\b(cause|effect|consequence|impact|influence|relationship between)\b',
    r'\b(theory|framework|model|approach|strategy|methodology)\b',
    r'\b(opinion|perspective|view|stance|argument|critique|analysis)\b',
]

_MATH = [
    r'\b(prove|proof|calculate|solve|compute|equation|integral|derivative|limit)\b',
    r'\b(matrix|vector|tensor|algebra|calculus|statistics|probability|theorem)\b',
    r'[∑∫∂∇≤≥±×÷√∞∈∉∀∃]',
    r'\b(formula|expression|simplify|factor|expand|differentiate|integrate)\b',
    r'\$\$.+?\$\$|\$.+?\$',  # LaTeX math
]

_CREATIVE = [
    r'\b(write|story|poem|creative|imagine|fiction|essay|narrative|character)\b',
    r'\b(generate|create|compose|draft|brainstorm|invent|design)\b',
    r'\b(blog|article|letter|email|script|screenplay|song|lyrics)\b',
]

_VISION = [
    r'\b(image|photo|picture|screenshot|diagram|chart|graph|figure)\b',
    r'\b(what is in|describe this|analyze this image|read this|ocr)\b',
]

_FAST = [
    r'^(what is|who is|when did|where is|translate|define|how do you say)\b',
    r'^(what\'s|who\'s|where\'s|when\'s)\b',
    r'\b(in one sentence|briefly|tldr|short answer|quick)\b',
]

_LONG_DOC_THRESHOLD = 4000


def classify(message: str, has_attachments: bool = False) -> RouterDecision:
    msg_lower = message.lower()

    scores = {
        "coding":    sum(1 for p in _CODING    if re.search(p, msg_lower)),
        "reasoning": sum(1 for p in _REASONING if re.search(p, msg_lower)),
        "math":      sum(1 for p in _MATH      if re.search(p, message)),  # keep case for LaTeX
        "creative":  sum(1 for p in _CREATIVE  if re.search(p, msg_lower)),
        "vision":    sum(1 for p in _VISION    if re.search(p, msg_lower)),
        "fast":      sum(1 for p in _FAST      if re.search(p, msg_lower)),
    }

    has_image = has_attachments  # caller knows attachment type
    is_long_doc = len(message) > _LONG_DOC_THRESHOLD
    estimated_tokens = len(message) // 4  # rough approximation

    # Determine primary intent
    if has_image:
        intent = "vision"
    elif is_long_doc:
        intent = "reasoning"  # long docs → Claude (200k context)
    elif scores["coding"] >= 2:
        intent = "coding"
    elif scores["math"] >= 1:
        intent = "math"
    elif scores["reasoning"] >= 2:
        intent = "reasoning"
    elif scores["creative"] >= 1:
        intent = "creative"
    elif scores["fast"] >= 1 and max(scores.values()) < 2:
        intent = "fast"
    else:
        # Tie-break by max score, default to reasoning
        top = max(scores, key=lambda k: scores[k])
        intent = top if scores[top] > 0 else "default"

    return RouterDecision(
        intent=intent,
        scores=scores,
        has_image=has_image,
        is_long_doc=is_long_doc,
        estimated_tokens=estimated_tokens,
    )
