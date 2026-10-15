"""Math answer verification and background normalization for MAGE."""

import re

from agentcore.llm.LLMBackend import BaseLLM


# ── LLM-as-Judge prompt ─────────────────────────────────────────────
JUDGE_SYSTEM_PROMPT = (
    "You are a mathematical answer verifier. "
    "Your job is to determine whether a predicted answer is mathematically "
    "equivalent to a reference answer.\n\n"
    "Two answers should be considered EQUIVALENT (and hence CORRECT) if:\n"
    "  - They express the same mathematical statement with different "
    "notation, variable names, or algebraic forms.\n"
    "  - One answer gives a more detailed or complete explanation than "
    "the other, but the core mathematical claim is the same.\n"
    "  - The predicted answer paraphrases, restates, or elaborates on "
    "the reference answer without contradicting it.\n"
    "  - Both answers identify the same mathematical objects, conditions, "
    "or characterizations, even if they use different wording or "
    "cite different (but logically equivalent) criteria.\n\n"
    "Focus on mathematical SUBSTANCE, not surface-level wording. "
    "Minor differences in presentation, level of detail, or style "
    "should NOT cause a CORRECT answer to be marked INCORRECT.\n\n"
    "Respond with EXACTLY one of:\n"
    "  CORRECT — if the predicted answer is mathematically equivalent "
    "to the reference (same core claim, possibly different wording).\n"
    "  INCORRECT — if the predicted answer is mathematically wrong or "
    "contradicts the reference answer.\n"
    "After your verdict, give a one-line explanation."
)

JUDGE_USER_TEMPLATE = (
    "### Background\n{background}\n\n"
    "### Question Context\n{question}\n\n"
    "### Reference Answer\n{reference}\n\n"
    "### Predicted Answer\n{predicted}\n\n"
    "Is the predicted answer mathematically equivalent to the reference "
    "answer? Focus on whether the core mathematical claim is the same, "
    "not on differences in wording, notation, or level of detail. "
    "Use the background to interpret definitions, notation, and assumptions."
)


class MathEnvironment:
    """Task-local judge; MAGE owns the reasoning state and history."""

    def __init__(self, judge_llm: BaseLLM = None):
        self.judge_llm = judge_llm
        self._judge_cache = {}

    # LaTeX preamble lines that carry no mathematical content
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

    @classmethod
    def _strip_latex_preamble(cls, text: str) -> str:
        """Remove LaTeX macro definitions and comments from background.

        Filters out lines starting with \\newcommand, \\renewcommand,
        \\DeclareMathOperator, \\DeclareMathAlphabet, \\usepackage,
        \\def, or pure-comment lines (%). Preserves all mathematical
        content such as definitions, theorems, etc.
        """
        if not text:
            return text
        filtered = []
        for line in text.split("\n"):
            if cls._PREAMBLE_PATTERNS.match(line):
                continue
            filtered.append(line)
        # Collapse runs of blank lines into a single blank line
        result = re.sub(r"\n{3,}", "\n\n", "\n".join(filtered))
        return result.strip()

    def judge_answer(
        self, question: str, background: str, predicted: str, reference: str
    ) -> tuple[bool, str]:
        """
        Use LLM-as-Judge to verify mathematical equivalence.
        Falls back to simple string matching if the judge is unavailable.
        """
        if self.judge_llm is None:
            return self._simple_verify(reference, predicted), "simple string match"

        prompt = JUDGE_USER_TEMPLATE.format(
            background=background,
            reference=reference,
            predicted=predicted,
            question=question,
        )
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
            print(f"[MathEnv] LLM judge failed: {exc}, falling back to string match")
            return self._simple_verify(reference, predicted), f"fallback: {exc}"

    @staticmethod
    def _simple_verify(reference: str, predicted: str) -> bool:
        """Basic string similarity fallback for answer verification."""
        def normalize(s: str) -> str:
            s = s.strip().lower()
            # Remove common LaTeX wrappers
            s = re.sub(r"^\$+|\$+$", "", s)
            s = re.sub(r"\\(?:left|right|big|Big|bigg|Bigg)", "", s)
            s = re.sub(r"\s+", " ", s)
            return s

        ref_norm = normalize(reference)
        pred_norm = normalize(predicted)
        return ref_norm == pred_norm
