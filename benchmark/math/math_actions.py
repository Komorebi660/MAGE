"""
Actions for Math Formal Reasoning benchmark.

Provides domain-specific tools:
  - SymbolicReasoning: formal derivation with optional SymPy verification
  - CodeExecutor: execute Python/SymPy code for computation
"""

import re
import subprocess
import sys

from agentcore.actions import BaseAction


def _try_parse_latex(latex_str: str):
    """Attempt to parse a LaTeX string into a SymPy expression.
    Returns (expr, None) on success, or (None, error_msg) on failure."""
    try:
        from sympy.parsing.latex import parse_latex
        # Clean up common LaTeX artifacts
        s = latex_str.strip().strip("$")
        # Remove display math delimiters
        s = re.sub(r"^\\\[|\\\]$", "", s).strip()
        s = re.sub(r"^\\begin\{.*?\}|\\end\{.*?\}$", "", s).strip()
        if not s:
            return None, "empty expression"
        expr = parse_latex(s)
        return expr, None
    except Exception as e:
        return None, str(e)


def _sympy_simplify(expr):
    """Simplify a SymPy expression, return (result_expr, latex_str)."""
    from sympy import simplify, latex
    result = simplify(expr)
    return result, latex(result)


def _sympy_expand(expr):
    from sympy import expand, latex
    result = expand(expr)
    return result, latex(result)


def _sympy_factor(expr):
    from sympy import factor, latex
    result = factor(expr)
    return result, latex(result)


# Map of supported operations
_OPERATIONS = {
    "simplify": _sympy_simplify,
    "expand": _sympy_expand,
    "factor": _sympy_factor,
}


class SymbolicReasoningAction(BaseAction):
    """
    Symbolic reasoning tool with optional SymPy-powered verification.

    The agent writes a reasoning step AND can optionally ask SymPy to
    verify or simplify expressions. This makes it a real tool that
    provides computational feedback, not just a Think action.

    Usage modes:
      1. Reasoning only: provide 'reasoning' → records the derivation step
      2. Reasoning + verify expression: provide 'reasoning' + 'expr' → also
         simplifies the expression via SymPy and returns the result
      3. Reasoning + check equality: provide 'reasoning' + 'expr' + 'expected'
         → checks if expr equals expected via SymPy
      4. Reasoning + operation: provide 'reasoning' + 'expr' + 'operation'
         → applies the operation (simplify/expand/factor) and returns result
    """

    def __init__(self):
        super().__init__(
            action_name="SymbolicReasoning",
            action_desc=(
                "Perform a step of symbolic mathematical reasoning with optional SymPy verification. "
                "Write your derivation in 'reasoning'. Optionally provide a LaTeX expression in 'expr' "
                "to have SymPy simplify/verify it. You can also set 'operation' to 'simplify', 'expand', "
                "or 'factor', or provide 'expected' to check equality between 'expr' and 'expected'."
            ),
            params_doc={
                "reasoning": (
                    "Your mathematical reasoning step. Include the logic, "
                    "formulas, and derivations in LaTeX format. (Required)"
                ),
                "expr": (
                    "A LaTeX expression to verify or simplify with SymPy. (Optional) "
                    "Example: '\\\\frac{x^2-1}{x-1}'"
                ),
                "expected": (
                    "A LaTeX expression that 'expr' should equal. If provided, "
                    "SymPy will check whether expr == expected. (Optional)"
                ),
                "operation": (
                    "Operation to apply to 'expr': 'simplify' (default), "
                    "'expand', or 'factor'. (Optional)"
                ),
            },
        )

    def __call__(
        self,
        reasoning: str = "",
        expr: str = "",
        expected: str = "",
        operation: str = "simplify",
    ) -> str:
        if not reasoning or not reasoning.strip():
            return "Error: Please provide non-empty reasoning content."

        parts = ["Reasoning recorded."]

        # If no expression to verify, just record reasoning
        if not expr or not expr.strip():
            return "OK. " + parts[0]

        # Parse the expression
        parsed_expr, err = _try_parse_latex(expr)
        if parsed_expr is None:
            parts.append(
                f"[SymPy] Could not parse expr: {err}. "
                "Reasoning is still recorded. Consider using CodeExecutor for complex expressions."
            )
            return " ".join(parts)

        # Mode 1: Check equality with expected
        if expected and expected.strip():
            parsed_expected, err2 = _try_parse_latex(expected)
            if parsed_expected is None:
                parts.append(f"[SymPy] Could not parse expected: {err2}.")
                return " ".join(parts)
            try:
                from sympy import simplify
                difference = simplify(parsed_expr - parsed_expected)
                if difference == 0 or difference.is_zero:
                    parts.append(f"[SymPy] ✓ Verified: expr equals expected.")
                else:
                    simp_expr = _sympy_simplify(parsed_expr)
                    simp_expected = _sympy_simplify(parsed_expected)
                    parts.append(
                        f"[SymPy] ✗ NOT equal. "
                        f"expr simplifies to: ${simp_expr[1]}$, "
                        f"expected simplifies to: ${simp_expected[1]}$."
                    )
            except Exception as e:
                parts.append(f"[SymPy] Equality check failed: {e}")
            return " ".join(parts)

        # Mode 2: Apply operation to expression
        op_func = _OPERATIONS.get(operation, _sympy_simplify)
        try:
            result_expr, result_latex = op_func(parsed_expr)
            parts.append(
                f"[SymPy] {operation}(${expr.strip().strip('$')}$) = ${result_latex}$"
            )
        except Exception as e:
            parts.append(f"[SymPy] {operation} failed: {e}")

        return " ".join(parts)


class CodeExecutorAction(BaseAction):
    """
    Execute Python code for numerical or symbolic computation.

    Supports SymPy, numpy, and standard math libraries.
    Returns stdout output or error messages.
    """

    # Limit execution time and output
    MAX_OUTPUT_LEN = 4096
    TIMEOUT_SECONDS = 120

    @classmethod
    def _limit_output(cls, output: str) -> str:
        if len(output) > cls.MAX_OUTPUT_LEN:
            return output[: cls.MAX_OUTPUT_LEN] + "\n... [output truncated]"
        return output

    @classmethod
    def _format_error(cls, message: str, detail: str = "") -> str:
        detail = cls._limit_output(detail.strip())
        if detail:
            return f"Code Error:\n{message}\n{detail}"
        return f"Code Error:\n{message}"

    def __init__(self):
        super().__init__(
            action_name="CodeExecutor",
            action_desc=(
                "Execute Python code for mathematical computation. "
                "You can use sympy for symbolic math, numpy for numerical "
                "computation, and math for basic operations. "
                "The code output (stdout) will be returned."
            ),
            params_doc={
                "code": (
                    "Python code to execute. Use print() to output results. "
                    "Available libraries: sympy, numpy, math, fractions."
                )
            },
        )

    def __call__(self, code: str = "") -> str:
        if not code or not code.strip():
            return "Error: Please provide non-empty code to execute."

        runner_code = f"""
import contextlib
import io
import sys
import traceback

sys.path[:] = {list(sys.path)!r}
code = {code!r}
stdout_capture = io.StringIO()
safe_globals = {{"__builtins__": __builtins__}}

for imp in [
    "import math",
    "from fractions import Fraction",
    "import sympy",
    "import numpy as np",
]:
    try:
        exec(imp, safe_globals)
    except ImportError:
        pass

try:
    with contextlib.redirect_stdout(stdout_capture):
        exec(compile(code, "<agent-code>", "exec"), safe_globals)
except SystemExit as exc:
    captured = stdout_capture.getvalue()
    if captured:
        print(captured, end="")
    if exc.code is None or exc.code == 0:
        raise
    print(f"SystemExit: {{exc.code}}", file=sys.stderr)
    raise
except BaseException:
    captured = stdout_capture.getvalue()
    if captured:
        print(captured, end="")
    traceback.print_exc()
    raise SystemExit(1)

print(stdout_capture.getvalue(), end="")
"""

        try:
            completed = subprocess.run(
                [sys.executable, "-c", runner_code],
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + (exc.stderr or "")
            return self._format_error(
                f"code execution timed out after {self.TIMEOUT_SECONDS} seconds",
                output,
            )
        except OSError as exc:
            return self._format_error("failed to start code execution", str(exc))

        output = completed.stdout or ""
        error_output = completed.stderr or ""
        if completed.returncode != 0:
            detail = (output + "\n" + error_output).strip()
            return self._format_error(
                f"code execution failed with exit code {completed.returncode}",
                detail,
            )

        if not output.strip():
            return "Code executed successfully (no output). Use print() to display results."
        output = self._limit_output(output)
        return f"Code Output:\n{output}"
