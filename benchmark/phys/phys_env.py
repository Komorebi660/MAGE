"""Physics answer verification and background normalization for MAGE."""

import re

from agentcore.llm.LLMBackend import BaseLLM


JUDGE_SYSTEM_PROMPT = (
    "You are a physics answer verifier. "
    "Your job is to determine whether a predicted answer is physically "
    "consistent with a reference answer.\n\n"
    "Treat answers as CORRECT when they express the same physical result, "
    "law, dependency, scaling, or derived quantity even if the notation, "
    "gauge, coordinate system, or algebraic form differs.\n"
    "Focus on physical substance rather than wording. Check whether:\n"
    "  - the same governing law or derived formula is identified\n"
    "  - assumptions, approximations, or boundary conditions are compatible\n"
    "  - units, dimensions, and sign conventions are consistent\n"
    "  - the prediction contradicts the reference in direction, scaling, or magnitude\n\n"
    "Respond with EXACTLY one of:\n"
    "  CORRECT — if the predicted answer matches the reference in physical content.\n"
    "  INCORRECT — if the predicted answer is physically wrong, inconsistent, or contradictory.\n"
    "After your verdict, give a one-line explanation."
)

JUDGE_USER_TEMPLATE = (
    "### Background\n{background}\n\n"
    "### Question Context\n{question}\n\n"
    "### Reference Answer\n{reference}\n\n"
    "### Predicted Answer\n{predicted}\n\n"
    "Is the predicted answer physically consistent with the reference answer? "
    "Focus on whether the same physical claim, formula, law, or derived quantity "
    "is expressed, not on wording differences alone. "
    "Use the background to interpret definitions, conventions, and assumptions."
)


class PhysEnvironment:
    """Task-local judge; MAGE owns the reasoning state and history."""

    _PREAMBLE_PATTERNS = re.compile(
        r"^\s*(?:"
        r"\\(?:newcommand|renewcommand|DeclareMathOperator|DeclareMathAlphabet"
        r"|usepackage|def\s)\\?"
        r"|\\newcommand\b"
        r"|%"
        r"|LaTeX\s+commands?\s*:"
        r")",
        re.IGNORECASE,
    )

    def __init__(self, judge_llm: BaseLLM = None):
        self.judge_llm = judge_llm
        self._judge_cache = {}

    @classmethod
    def _strip_latex_preamble(cls, text: str) -> str:
        if not text:
            return text
        filtered = []
        for line in text.split("\n"):
            if cls._PREAMBLE_PATTERNS.match(line):
                continue
            filtered.append(line)
        return re.sub(r"\n{3,}", "\n\n", "\n".join(filtered)).strip()

    def judge_answer(self, question: str, background: str, predicted: str, reference: str) -> tuple[bool, str]:
        if self.judge_llm is None:
            return self._simple_verify(reference, predicted), "simple string match"

        prompt = JUDGE_USER_TEMPLATE.format(background=background, reference=reference, predicted=predicted, question=question)
        judge = self.judge_llm
        cache_key = (
            judge.llm_name, judge.base_url,
            judge.temperature, judge.top_p, judge.seed, judge.enable_thinking,
            JUDGE_SYSTEM_PROMPT, prompt,
        )
        if cache_key in self._judge_cache:
            return self._judge_cache[cache_key]

        try:
            response = judge.run(prompt, system_prompt=JUDGE_SYSTEM_PROMPT)
            response_clean = response.strip()
            response_normalized = re.sub(r"[^\w\s]", "", response_clean).strip().upper()
            correct = "CORRECT" in response_normalized and "INCORRECT" not in response_normalized
            result = correct, response_clean
            # Cache only one distinct verdict keyword; repeated occurrences are allowed.
            # Missing or mixed keywords skip caching without changing the score above.
            verdicts = set(re.findall(r"\b(?:CORRECT|INCORRECT)\b", response_normalized))
            if len(verdicts) == 1:
                self._judge_cache[cache_key] = result
            return result
        except Exception as exc:
            print(f"[PhysEnv] LLM judge failed: {exc}, falling back to string match")
            return self._simple_verify(reference, predicted), f"fallback: {exc}"

    @staticmethod
    def _simple_verify(reference: str, predicted: str) -> bool:
        def normalize(text: str) -> str:
            text = text.strip().lower()
            text = re.sub(r"^\$+|\$+$", "", text)
            text = re.sub(r"\\(?:left|right|big|Big|bigg|Bigg)", "", text)
            text = re.sub(r"\s+", " ", text)
            return text

        return normalize(reference) == normalize(predicted)