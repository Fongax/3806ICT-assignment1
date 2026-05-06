"""
Restricted TPTP FOF benchmark runner for 3806ICT Assignment 1.

Place this file next to prover.py.

Expected folder layout:

3806ICT-assignment1/
├── prover.py
├── tptp_runner.py
├── tptp_benchmarks/
│   ├── KRS018+1.p
│   ├── MGT019+2.p
│   ├── PUZ001+1.p
│   ├── SYN000+1.p
│   ├── SYN075+1.p
│   └── KRS063+1.p
└── results/

Run:
    python tptp_runner.py

Optional:
    python tptp_runner.py --folder tptp_benchmarks --results results/tptp_results.csv

Important limitation:
This is a restricted TPTP FOF reader, not a full TPTP implementation.
It supports selected FOF syntax only and converts equality into the ordinary
predicate eq(x,y), because the current LK' prover does not implement equality rules.
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from prover import (
    And,
    BaselineProver,
    Bottom,
    Const,
    Exists,
    Formula,
    Forall,
    Func,
    Imp,
    ImprovedProver,
    Not,
    Or,
    Pred,
    Term,
    Top,
    Var,
)


class TPTPParseError(Exception):
    pass


TPTP_TOKEN_RE = re.compile(
    r"\s*(<=>|<~>|=>|<=|!=|[!\?\[\]:(),&|~=]|\$true|\$false|'(?:\\.|[^\\'])*'|[A-Za-z][A-Za-z0-9_]*|[0-9]+)"
)


def strip_tptp_comments(text: str) -> str:
    """Remove TPTP line comments and block comments."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    lines = []
    for line in text.splitlines():
        if "%" in line:
            line = line.split("%", 1)[0]
        lines.append(line)
    return "\n".join(lines)


def split_tptp_statements(text: str) -> List[str]:
    """Split a TPTP file into top-level statements ending in a period."""
    clean = strip_tptp_comments(text)
    statements: List[str] = []
    buf: List[str] = []
    paren_depth = 0
    bracket_depth = 0
    in_quote = False
    escape = False

    for ch in clean:
        buf.append(ch)

        if in_quote:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "'":
                in_quote = False
            continue

        if ch == "'":
            in_quote = True
        elif ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth -= 1
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth -= 1
        elif ch == "." and paren_depth == 0 and bracket_depth == 0:
            statement = "".join(buf).strip()
            if statement:
                statements.append(statement[:-1].strip())
            buf = []

    return statements


def split_top_level_commas(text: str) -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    paren_depth = 0
    bracket_depth = 0
    in_quote = False
    escape = False

    for ch in text:
        if in_quote:
            buf.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "'":
                in_quote = False
            continue

        if ch == "'":
            in_quote = True
            buf.append(ch)
        elif ch == "(":
            paren_depth += 1
            buf.append(ch)
        elif ch == ")":
            paren_depth -= 1
            buf.append(ch)
        elif ch == "[":
            bracket_depth += 1
            buf.append(ch)
        elif ch == "]":
            bracket_depth -= 1
            buf.append(ch)
        elif ch == "," and paren_depth == 0 and bracket_depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)

    if buf:
        parts.append("".join(buf).strip())
    return parts


def clean_tptp_name(name: str) -> str:
    """Convert quoted TPTP names into simple internal names."""
    name = name.strip()
    if name.startswith("'") and name.endswith("'"):
        name = name[1:-1]
    name = name.replace("\\'", "'").replace("\\\\", "\\")
    safe = re.sub(r"[^A-Za-z0-9_]", "_", name)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "quoted_symbol"


class TPTPParser:
    """
    Restricted TPTP FOF parser.

    Supported fragment:
    - fof(name, role, formula).
    - roles: axiom, hypothesis, conjecture, definition, lemma, plain
    - connectives: ~, &, |, =>, <=, <=>, <~>
    - quantifiers: ! [X] : F and ? [X] : F
    - equality and inequality translated into eq(x,y) and not eq(x,y)

    Unsupported:
    - include expansion
    - typed TPTP
    - arithmetic
    - built-in equality reasoning beyond treating equality as predicate eq
    """

    def __init__(self, text: str):
        self.tokens = self._tokenise(text)
        self.i = 0

    def _tokenise(self, text: str) -> List[str]:
        tokens: List[str] = []
        pos = 0
        while pos < len(text):
            if text[pos].isspace():
                pos += 1
                continue
            match = TPTP_TOKEN_RE.match(text, pos)
            if not match:
                raise TPTPParseError(f"Unsupported TPTP token near: {text[pos:pos + 40]!r}")
            tokens.append(match.group(1))
            pos = match.end()
        return tokens

    def peek(self) -> Optional[str]:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def pop(self, expected: Optional[str] = None) -> str:
        tok = self.peek()
        if tok is None:
            raise TPTPParseError("Unexpected end of TPTP formula")
        if expected is not None and tok != expected:
            raise TPTPParseError(f"Expected {expected!r}, got {tok!r}")
        self.i += 1
        return tok

    def parse(self) -> Formula:
        formula = self.parse_iff()
        if self.peek() is not None:
            raise TPTPParseError(f"Unexpected token: {self.peek()}")
        return formula

    def parse_iff(self) -> Formula:
        left = self.parse_imp()
        while self.peek() in {"<=>", "<~>"}:
            op = self.pop()
            right = self.parse_imp()
            iff_formula = And(Imp(left, right), Imp(right, left))
            if op == "<~>":
                left = Not(iff_formula)
            else:
                left = iff_formula
        return left

    def parse_imp(self) -> Formula:
        left = self.parse_or()
        if self.peek() in {"=>", "<="}:
            op = self.pop()
            right = self.parse_imp()
            if op == "=>":
                return Imp(left, right)
            return Imp(right, left)
        return left

    def parse_or(self) -> Formula:
        f = self.parse_and()
        while self.peek() == "|":
            self.pop("|")
            f = Or(f, self.parse_and())
        return f

    def parse_and(self) -> Formula:
        f = self.parse_unary()
        while self.peek() == "&":
            self.pop("&")
            f = And(f, self.parse_unary())
        return f

    def parse_unary(self) -> Formula:
        tok = self.peek()
        if tok == "~":
            self.pop("~")
            return Not(self.parse_unary())

        if tok in {"!", "?"}:
            quantifier = self.pop()
            self.pop("[")
            variables = [clean_tptp_name(self.pop())]
            while self.peek() == ",":
                self.pop(",")
                variables.append(clean_tptp_name(self.pop()))
            self.pop("]")
            self.pop(":")
            body = self.parse_iff()
            for var in reversed(variables):
                body = Forall(var, body) if quantifier == "!" else Exists(var, body)
            return body

        return self.parse_atom()

    def parse_atom(self) -> Formula:
        if self.peek() == "(":
            self.pop("(")
            f = self.parse_iff()
            self.pop(")")
            return f

        if self.peek() == "$true":
            self.pop()
            return Top()

        if self.peek() == "$false":
            self.pop()
            return Bottom()

        left_term = self.parse_term()

        if self.peek() in {"=", "!="}:
            op = self.pop()
            right_term = self.parse_term()
            equality = Pred("eq", (left_term, right_term))
            return equality if op == "=" else Not(equality)

        if isinstance(left_term, Func):
            return Pred(left_term.name, left_term.args)
        if isinstance(left_term, Const):
            return Pred(left_term.name)
        if isinstance(left_term, Var):
            return Pred(left_term.name)

        raise TPTPParseError("Expected TPTP atom")

    def parse_term(self) -> Term:
        tok = self.pop()
        name = clean_tptp_name(tok)

        if self.peek() == "(":
            self.pop("(")
            args: List[Term] = []
            if self.peek() != ")":
                args.append(self.parse_term())
                while self.peek() == ",":
                    self.pop(",")
                    args.append(self.parse_term())
            self.pop(")")
            return Func(name, tuple(args))

        if tok and tok[0].isupper():
            return Var(name)

        return Const(name)


def parse_tptp_formula(text: str) -> Formula:
    return TPTPParser(text).parse()


def parse_tptp_status(text: str) -> str:
    match = re.search(r"%\s*Status\s*:\s*([A-Za-z_]+)", text)
    return match.group(1) if match else "unknown"


def parse_fof_statement(statement: str) -> Optional[Tuple[str, str, Formula]]:
    if not statement.startswith("fof"):
        return None

    open_index = statement.find("(")
    close_index = statement.rfind(")")
    if open_index < 0 or close_index < open_index:
        raise TPTPParseError(f"Malformed fof statement: {statement[:80]}")

    inner = statement[open_index + 1:close_index]
    parts = split_top_level_commas(inner)
    if len(parts) < 3:
        raise TPTPParseError(f"Expected fof(name, role, formula): {statement[:80]}")

    name = clean_tptp_name(parts[0])
    role = clean_tptp_name(parts[1]).lower()
    formula = parse_tptp_formula(parts[2])
    return name, role, formula


def conjoin_formulae(formulae: Sequence[Formula]) -> Formula:
    if not formulae:
        return Top()
    result = formulae[0]
    for f in formulae[1:]:
        result = And(result, f)
    return result


@dataclass
class TPTPProblem:
    filename: str
    status: str
    mode: str
    formula_count: int
    goal: Formula


def parse_tptp_problem(path: Path) -> TPTPProblem:
    text = path.read_text(encoding="utf-8")
    status = parse_tptp_status(text)

    axioms: List[Formula] = []
    conjectures: List[Formula] = []
    formula_count = 0

    for statement in split_tptp_statements(text):
        parsed = parse_fof_statement(statement)
        if parsed is None:
            # Restricted parser ignores include(...) and non-fof statements.
            continue

        _name, role, formula = parsed
        formula_count += 1

        if role in {"axiom", "hypothesis", "definition", "lemma", "plain"}:
            axioms.append(formula)
        elif role == "conjecture":
            conjectures.append(formula)
        elif role == "negated_conjecture":
            axioms.append(formula)
        else:
            axioms.append(formula)

    if conjectures:
        goal = Imp(conjoin_formulae(axioms), conjoin_formulae(conjectures))
        mode = "entailment"
    else:
        # For satisfiable/unsatisfiable files with no conjecture, test whether
        # the axioms imply contradiction.
        goal = Imp(conjoin_formulae(axioms), Bottom())
        mode = "unsat_check"

    return TPTPProblem(path.name, status, mode, formula_count, goal)


def run_one_problem(
    prover_name: str,
    prover,
    problem: TPTPProblem,
) -> Dict[str, object]:
    try:
        start = time.perf_counter()
        result = prover.prove(problem.goal)
        elapsed = time.perf_counter() - start
        return {
            "dataset": problem.filename,
            "prover": prover_name,
            "result_status": result.status,
            "nodes_expanded": result.nodes_expanded,
            "elapsed_seconds": round(elapsed, 6),
            "tptp_status": problem.status,
            "tptp_mode": problem.mode,
            "tptp_formula_count": problem.formula_count,
            "error": "",
        }
    except RecursionError as exc:
        return {
            "dataset": problem.filename,
            "prover": prover_name,
            "result_status": "recursion_error",
            "nodes_expanded": getattr(prover, "nodes_expanded", 0),
            "elapsed_seconds": "",
            "tptp_status": problem.status,
            "tptp_mode": problem.mode,
            "tptp_formula_count": problem.formula_count,
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "dataset": problem.filename,
            "prover": prover_name,
            "result_status": "runtime_error",
            "nodes_expanded": getattr(prover, "nodes_expanded", 0),
            "elapsed_seconds": "",
            "tptp_status": problem.status,
            "tptp_mode": problem.mode,
            "tptp_formula_count": problem.formula_count,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_tptp_benchmarks(folder: Path, results_path: Path, max_nodes: int, max_seconds: float) -> None:
    rows: List[Dict[str, object]] = []

    files = sorted(folder.glob("*.p"))
    if not files:
        raise FileNotFoundError(f"No .p files found in {folder}")

    for path in files:
        try:
            problem = parse_tptp_problem(path)
            rows.append(run_one_problem("baseline", BaselineProver(max_nodes=max_nodes, max_seconds=max_seconds), problem))
            rows.append(run_one_problem("improved", ImprovedProver(max_nodes=max_nodes, max_seconds=max_seconds), problem))
        except Exception as exc:
            status = parse_tptp_status(path.read_text(encoding="utf-8"))
            for prover_name in ["baseline", "improved"]:
                rows.append({
                    "dataset": path.name,
                    "prover": prover_name,
                    "result_status": "tptp_parse_error",
                    "nodes_expanded": 0,
                    "elapsed_seconds": "",
                    "tptp_status": status,
                    "tptp_mode": "unknown",
                    "tptp_formula_count": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                })

    results_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "prover",
        "result_status",
        "nodes_expanded",
        "elapsed_seconds",
        "tptp_status",
        "tptp_mode",
        "tptp_formula_count",
        "error",
    ]

    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote TPTP benchmark results to: {results_path}")
    print_summary(rows)


def print_summary(rows: Sequence[Dict[str, object]]) -> None:
    print()
    print("dataset, prover, result_status, nodes_expanded, elapsed_seconds, tptp_status, tptp_mode, error")
    for row in rows:
        print(
            f"{row['dataset']}, {row['prover']}, {row['result_status']}, "
            f"{row['nodes_expanded']}, {row['elapsed_seconds']}, "
            f"{row['tptp_status']}, {row['tptp_mode']}, {row['error']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Restricted TPTP FOF runner for the 3806ICT prover")
    parser.add_argument("--folder", default="tptp_benchmarks", help="folder containing TPTP .p files")
    parser.add_argument("--results", default="results/tptp_results.csv", help="output CSV path")
    parser.add_argument("--max-nodes", type=int, default=10000)
    parser.add_argument("--max-seconds", type=float, default=5.0)
    args = parser.parse_args()

    run_tptp_benchmarks(
        folder=Path(args.folder),
        results_path=Path(args.results),
        max_nodes=args.max_nodes,
        max_seconds=args.max_seconds,
    )


if __name__ == "__main__":
    main()
