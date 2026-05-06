"""
3806ICT Assignment 1 implementation.

Purpose:
- Parse first-order logic formulae from text.
- Represent formulae and terms as Python data classes.
- Implement a baseline backward proof-search strategy based on Algorithm 2.
- Compare it with an improved proof-search control strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple, Union
import argparse
import csv
import re
import time


# ---------------------------------------------------------------------------
# Term representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class Var:
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, order=True)
class Const:
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, order=True)
class Func:
    name: str
    args: Tuple["Term", ...]

    def __str__(self) -> str:
        return f"{self.name}({', '.join(map(str, self.args))})"


Term = Union[Var, Const, Func]


def term_vars(t: Term) -> Set[str]:
    if isinstance(t, Var):
        return {t.name}
    if isinstance(t, Const):
        return set()
    if isinstance(t, Func):
        out: Set[str] = set()
        for a in t.args:
            out |= term_vars(a)
        return out
    raise TypeError(t)


def subst_term(t: Term, var: str, replacement: Term) -> Term:
    if isinstance(t, Var):
        return replacement if t.name == var else t
    if isinstance(t, Const):
        return t
    if isinstance(t, Func):
        return Func(t.name, tuple(subst_term(a, var, replacement) for a in t.args))
    raise TypeError(t)


# ---------------------------------------------------------------------------
# Formula representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class Top:
    def __str__(self) -> str:
        return "⊤"


@dataclass(frozen=True, order=True)
class Bottom:
    def __str__(self) -> str:
        return "⊥"


@dataclass(frozen=True, order=True)
class Pred:
    name: str
    args: Tuple[Term, ...] = ()

    def __str__(self) -> str:
        if not self.args:
            return self.name
        return f"{self.name}({', '.join(map(str, self.args))})"


@dataclass(frozen=True, order=True)
class Not:
    body: "Formula"

    def __str__(self) -> str:
        return f"¬{parenthesise(self.body)}"


@dataclass(frozen=True, order=True)
class And:
    left: "Formula"
    right: "Formula"

    def __str__(self) -> str:
        return f"({self.left} ∧ {self.right})"


@dataclass(frozen=True, order=True)
class Or:
    left: "Formula"
    right: "Formula"

    def __str__(self) -> str:
        return f"({self.left} ∨ {self.right})"


@dataclass(frozen=True, order=True)
class Imp:
    left: "Formula"
    right: "Formula"

    def __str__(self) -> str:
        return f"({self.left} → {self.right})"


@dataclass(frozen=True, order=True)
class Forall:
    var: str
    body: "Formula"

    def __str__(self) -> str:
        return f"∀{self.var}. {self.body}"


@dataclass(frozen=True, order=True)
class Exists:
    var: str
    body: "Formula"

    def __str__(self) -> str:
        return f"∃{self.var}. {self.body}"


Formula = Union[Top, Bottom, Pred, Not, And, Or, Imp, Forall, Exists]


def parenthesise(f: Formula) -> str:
    if isinstance(f, (Pred, Top, Bottom)):
        return str(f)
    return f"({f})"


def subst_formula(f: Formula, var: str, replacement: Term) -> Formula:
    """Capture-avoiding enough for this assignment if fresh constants are used."""
    if isinstance(f, (Top, Bottom)):
        return f
    if isinstance(f, Pred):
        return Pred(f.name, tuple(subst_term(a, var, replacement) for a in f.args))
    if isinstance(f, Not):
        return Not(subst_formula(f.body, var, replacement))
    if isinstance(f, And):
        return And(subst_formula(f.left, var, replacement), subst_formula(f.right, var, replacement))
    if isinstance(f, Or):
        return Or(subst_formula(f.left, var, replacement), subst_formula(f.right, var, replacement))
    if isinstance(f, Imp):
        return Imp(subst_formula(f.left, var, replacement), subst_formula(f.right, var, replacement))
    if isinstance(f, Forall):
        if f.var == var:
            return f
        return Forall(f.var, subst_formula(f.body, var, replacement))
    if isinstance(f, Exists):
        if f.var == var:
            return f
        return Exists(f.var, subst_formula(f.body, var, replacement))
    raise TypeError(f)


def formula_terms(f: Formula) -> Set[Term]:
    """Collect ground terms already occurring in a formula."""
    out: Set[Term] = set()

    def walk_term(t: Term) -> None:
        if isinstance(t, Const):
            out.add(t)
        elif isinstance(t, Func):
            if not term_vars(t):
                out.add(t)
            for a in t.args:
                walk_term(a)

    def walk_formula(g: Formula) -> None:
        if isinstance(g, Pred):
            for a in g.args:
                walk_term(a)
        elif isinstance(g, Not):
            walk_formula(g.body)
        elif isinstance(g, (And, Or, Imp)):
            walk_formula(g.left)
            walk_formula(g.right)
        elif isinstance(g, (Forall, Exists)):
            walk_formula(g.body)

    walk_formula(f)
    return out


# ---------------------------------------------------------------------------
# Sequent and proof-search data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sequent:
    left: FrozenSet[Formula]
    right: FrozenSet[Formula]

    def __str__(self) -> str:
        l = ", ".join(map(str, sorted(self.left, key=str))) or " "
        r = ", ".join(map(str, sorted(self.right, key=str))) or " "
        return f"{l} ⊢ {r}"

    @staticmethod
    def start(goal: Formula) -> "Sequent":
        return Sequent(frozenset(), frozenset({goal}))


@dataclass
class ProofNode:
    sequent: Sequent
    rule: str = "open"
    children: List["ProofNode"] = field(default_factory=list)
    closed: bool = False


@dataclass
class SearchResult:
    status: str
    elapsed: float
    nodes_expanded: int
    root: ProofNode


# ---------------------------------------------------------------------------
# Baseline prover
# ---------------------------------------------------------------------------

class BaselineProver:
    def __init__(self, max_nodes: int = 10_000, max_seconds: float = 5.0):
        self.max_nodes = max_nodes
        self.max_seconds = max_seconds
        self.nodes_expanded = 0
        self.fresh_counter = 0
        self.quantifier_history: Dict[Tuple[Sequent, Formula, str], Set[Term]] = {}
        self.start_time = 0.0

    def prove(self, goal: Formula) -> SearchResult:
        self.nodes_expanded = 0
        self.fresh_counter = 0
        self.quantifier_history.clear()
        self.start_time = time.perf_counter()

        root = ProofNode(Sequent.start(goal))
        status = "proved" if self._expand(root) else "failed"
        elapsed = time.perf_counter() - self.start_time
        if elapsed >= self.max_seconds or self.nodes_expanded >= self.max_nodes:
            status = "timeout"
        return SearchResult(status, elapsed, self.nodes_expanded, root)

    def _expand(self, node: ProofNode) -> bool:
        if time.perf_counter() - self.start_time > self.max_seconds:
            return False
        if self.nodes_expanded >= self.max_nodes:
            return False

        self.nodes_expanded += 1
        seq = node.sequent

        close_rule = self._closing_rule(seq)
        if close_rule:
            node.rule = close_rule
            node.closed = True
            return True

        step = self._next_baseline_step(seq)
        if step is None:
            node.rule = "stuck"
            node.closed = False
            return False

        rule_name, premises = step
        node.rule = rule_name
        node.children = [ProofNode(p) for p in premises]

        all_closed = True
        for child in node.children:
            if not self._expand(child):
                all_closed = False
        node.closed = all_closed
        return all_closed

    def _closing_rule(self, seq: Sequent) -> Optional[str]:
        if seq.left & seq.right:
            return "id"
        if any(isinstance(f, Top) for f in seq.right):
            return "⊤R"
        if any(isinstance(f, Bottom) for f in seq.left):
            return "⊥L"
        return None

    def _next_baseline_step(self, seq: Sequent) -> Optional[Tuple[str, List[Sequent]]]:
        for f in sorted(seq.left, key=str):
            if isinstance(f, And):
                return "∧L", [replace_left(seq, f, [f.left, f.right])]
            if isinstance(f, Not):
                return "¬L", [remove_left_add_right(seq, f, f.body)]
            if isinstance(f, Exists):
                c = self.fresh_const()
                return "∃L", [replace_left(seq, f, [subst_formula(f.body, f.var, c)])]

        for f in sorted(seq.right, key=str):
            if isinstance(f, Or):
                return "∨R", [replace_right(seq, f, [f.left, f.right])]
            if isinstance(f, Imp):
                return "→R", [Sequent(seq.left | frozenset({f.left}), (seq.right - frozenset({f})) | frozenset({f.right}))]
            if isinstance(f, Not):
                return "¬R", [remove_right_add_left(seq, f, f.body)]
            if isinstance(f, Forall):
                c = self.fresh_const()
                return "∀R", [replace_right(seq, f, [subst_formula(f.body, f.var, c)])]

        for f in sorted(seq.right, key=str):
            if isinstance(f, And):
                return "∧R", [replace_right(seq, f, [f.left]), replace_right(seq, f, [f.right])]

        for f in sorted(seq.left, key=str):
            if isinstance(f, Or):
                return "∨L", [replace_left(seq, f, [f.left]), replace_left(seq, f, [f.right])]
            if isinstance(f, Imp):
                return "→L", [
                    Sequent(seq.left - frozenset({f}), seq.right | frozenset({f.left})),
                    Sequent((seq.left - frozenset({f})) | frozenset({f.right}), seq.right),
                ]

        ground_terms = self._available_ground_terms(seq)

        for f in sorted(seq.left, key=str):
            if isinstance(f, Forall):
                t = self._next_unused_term(seq, f, "∀L", ground_terms)
                if t is None:
                    t = self.fresh_const()
                self._mark_used(seq, f, "∀L", t)
                instance = subst_formula(f.body, f.var, t)
                return "∀L", [Sequent(seq.left | frozenset({instance}), seq.right)]

        for f in sorted(seq.right, key=str):
            if isinstance(f, Exists):
                t = self._next_unused_term(seq, f, "∃R", ground_terms)
                if t is None:
                    t = self.fresh_const()
                self._mark_used(seq, f, "∃R", t)
                instance = subst_formula(f.body, f.var, t)
                return "∃R", [Sequent(seq.left, seq.right | frozenset({instance}))]

        return None

    def _available_ground_terms(self, seq: Sequent) -> List[Term]:
        terms: Set[Term] = set()
        for f in seq.left | seq.right:
            terms |= formula_terms(f)
        if not terms:
            terms.add(Const("c0"))
        return sorted(terms, key=str)

    def _next_unused_term(self, seq: Sequent, formula: Formula, rule: str, terms: List[Term]) -> Optional[Term]:
        used = self.quantifier_history.get((seq, formula, rule), set())
        for t in terms:
            if t not in used:
                return t
        return None

    def _mark_used(self, seq: Sequent, formula: Formula, rule: str, term: Term) -> None:
        key = (seq, formula, rule)
        self.quantifier_history.setdefault(key, set()).add(term)

    def fresh_const(self) -> Const:
        self.fresh_counter += 1
        return Const(f"c{self.fresh_counter}")


# ---------------------------------------------------------------------------
# Improved prover
# ---------------------------------------------------------------------------

class ImprovedProver(BaselineProver):
    """
    Improved proof-search control.

    The implementation records sequents encountered on the current proof-search
    path and stops a branch when the same sequent is reached again. It also avoids
    quantified-rule applications that do not add a new instance to the sequent.
    """

    def __init__(self, max_nodes: int = 10_000, max_seconds: float = 5.0):
        super().__init__(max_nodes=max_nodes, max_seconds=max_seconds)
        self.seen_results: Dict[Sequent, bool] = {}
        self.visiting: Set[Sequent] = set()

    def prove(self, goal: Formula) -> SearchResult:
        self.seen_results.clear()
        self.visiting.clear()
        return super().prove(goal)

    def _expand(self, node: ProofNode) -> bool:
        seq = node.sequent

        if seq in self.seen_results:
            node.rule = "repeated-sequent"
            node.closed = self.seen_results[seq]
            return self.seen_results[seq]

        if seq in self.visiting:
            node.rule = "loop-cut"
            node.closed = False
            self.seen_results[seq] = False
            return False

        self.visiting.add(seq)
        result = super()._expand(node)
        self.visiting.remove(seq)
        self.seen_results[seq] = result
        return result

    def _next_baseline_step(self, seq: Sequent) -> Optional[Tuple[str, List[Sequent]]]:
        for f in sorted(seq.left, key=str):
            if isinstance(f, And):
                return "∧L", [replace_left(seq, f, [f.left, f.right])]
            if isinstance(f, Not):
                return "¬L", [remove_left_add_right(seq, f, f.body)]
            if isinstance(f, Exists):
                c = self.fresh_const()
                return "∃L", [replace_left(seq, f, [subst_formula(f.body, f.var, c)])]

        for f in sorted(seq.right, key=str):
            if isinstance(f, Or):
                return "∨R", [replace_right(seq, f, [f.left, f.right])]
            if isinstance(f, Imp):
                return "→R", [Sequent(seq.left | frozenset({f.left}), (seq.right - frozenset({f})) | frozenset({f.right}))]
            if isinstance(f, Not):
                return "¬R", [remove_right_add_left(seq, f, f.body)]
            if isinstance(f, Forall):
                c = self.fresh_const()
                return "∀R", [replace_right(seq, f, [subst_formula(f.body, f.var, c)])]

        for f in sorted(seq.right, key=str):
            if isinstance(f, And):
                return "∧R", [replace_right(seq, f, [f.left]), replace_right(seq, f, [f.right])]

        for f in sorted(seq.left, key=str):
            if isinstance(f, Or):
                return "∨L", [replace_left(seq, f, [f.left]), replace_left(seq, f, [f.right])]
            if isinstance(f, Imp):
                return "→L", [
                    Sequent(seq.left - frozenset({f}), seq.right | frozenset({f.left})),
                    Sequent((seq.left - frozenset({f})) | frozenset({f.right}), seq.right),
                ]

        ground_terms = self._available_ground_terms(seq)

        for f in sorted(seq.left, key=str):
            if isinstance(f, Forall):
                for t in ground_terms:
                    instance = subst_formula(f.body, f.var, t)
                    if instance not in seq.left:
                        return "∀L-useful", [Sequent(seq.left | frozenset({instance}), seq.right)]

        for f in sorted(seq.right, key=str):
            if isinstance(f, Exists):
                for t in ground_terms:
                    instance = subst_formula(f.body, f.var, t)
                    if instance not in seq.right:
                        return "∃R-useful", [Sequent(seq.left, seq.right | frozenset({instance}))]

        for f in sorted(seq.left, key=str):
            if isinstance(f, Forall):
                c = self.fresh_const()
                instance = subst_formula(f.body, f.var, c)
                if instance not in seq.left:
                    return "∀L-fresh", [Sequent(seq.left | frozenset({instance}), seq.right)]

        for f in sorted(seq.right, key=str):
            if isinstance(f, Exists):
                c = self.fresh_const()
                instance = subst_formula(f.body, f.var, c)
                if instance not in seq.right:
                    return "∃R-fresh", [Sequent(seq.left, seq.right | frozenset({instance}))]

        return None


# ---------------------------------------------------------------------------
# Sequent update helpers
# ---------------------------------------------------------------------------

def replace_left(seq: Sequent, old: Formula, new_items: Iterable[Formula]) -> Sequent:
    return Sequent((seq.left - frozenset({old})) | frozenset(new_items), seq.right)


def replace_right(seq: Sequent, old: Formula, new_items: Iterable[Formula]) -> Sequent:
    return Sequent(seq.left, (seq.right - frozenset({old})) | frozenset(new_items))


def remove_left_add_right(seq: Sequent, old: Formula, moved: Formula) -> Sequent:
    return Sequent(seq.left - frozenset({old}), seq.right | frozenset({moved}))


def remove_right_add_left(seq: Sequent, old: Formula, moved: Formula) -> Sequent:
    return Sequent(seq.left | frozenset({moved}), seq.right - frozenset({old}))


# ---------------------------------------------------------------------------
# Text formula parser
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(
    r"\s*(->|=>|/\\|&&|\\/|\|\||~|!|not\b|forall\b|exists\b|True\b|False\b|[A-Za-z_][A-Za-z0-9_]*|[().,])"
)


class ParseError(Exception):
    pass


class Parser:
    """
    Small recursive-descent parser.

    Supported examples:
    - P(a)
    - forall x. P(x) -> Q(x)
    - exists x. P(x) /\\ Q(x)
    - not P(a)
    """

    def __init__(self, text: str):
        self.tokens = self._tokenise(text)
        self.i = 0

    def _tokenise(self, text: str) -> List[str]:
        tokens = TOKEN_RE.findall(text)
        compact = "".join(tokens).replace(" ", "")
        stripped = re.sub(r"\s+", "", text)
        if compact != stripped:
            raise ParseError(f"Unexpected character in: {text!r}")
        return tokens

    def peek(self) -> Optional[str]:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def pop(self, expected: Optional[str] = None) -> str:
        tok = self.peek()
        if tok is None:
            raise ParseError("Unexpected end of input")
        if expected is not None and tok != expected:
            raise ParseError(f"Expected {expected!r}, got {tok!r}")
        self.i += 1
        return tok

    def parse(self) -> Formula:
        f = self.parse_imp()
        if self.peek() is not None:
            raise ParseError(f"Unexpected token: {self.peek()}")
        return f

    def parse_imp(self) -> Formula:
        left = self.parse_or()
        if self.peek() in {"->", "=>"}:
            self.pop()
            right = self.parse_imp()
            return Imp(left, right)
        return left

    def parse_or(self) -> Formula:
        f = self.parse_and()
        while self.peek() in {"\\/", "||"}:
            self.pop()
            f = Or(f, self.parse_and())
        return f

    def parse_and(self) -> Formula:
        f = self.parse_not()
        while self.peek() in {"/\\", "&&"}:
            self.pop()
            f = And(f, self.parse_not())
        return f

    def parse_not(self) -> Formula:
        tok = self.peek()
        if tok in {"~", "!", "not"}:
            self.pop()
            return Not(self.parse_not())
        if tok == "forall":
            self.pop()
            var = self.pop()
            self.pop(".")
            return Forall(var, self.parse_imp())
        if tok == "exists":
            self.pop()
            var = self.pop()
            self.pop(".")
            return Exists(var, self.parse_imp())
        return self.parse_atom()

    def parse_atom(self) -> Formula:
        tok = self.peek()
        if tok == "(":
            self.pop("(")
            f = self.parse_imp()
            self.pop(")")
            return f
        if tok == "True":
            self.pop()
            return Top()
        if tok == "False":
            self.pop()
            return Bottom()
        if tok is None:
            raise ParseError("Expected formula")

        name = self.pop()
        if self.peek() == "(":
            self.pop("(")
            args: List[Term] = []
            if self.peek() != ")":
                args.append(self.parse_term())
                while self.peek() == ",":
                    self.pop(",")
                    args.append(self.parse_term())
            self.pop(")")
            return Pred(name, tuple(args))
        return Pred(name)

    def parse_term(self) -> Term:
        name = self.pop()
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

        if name in {"x", "y", "z", "u", "v", "w"}:
            return Var(name)
        return Const(name)


def parse_formula(text: str) -> Formula:
    return Parser(text).parse()


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

EXAMPLES = [
    "P(a) -> P(a)",
    "(P(a) && Q(a)) -> P(a)",
    "P(a) -> (P(a) || Q(a))",
    "(forall x. P(x)) -> P(a)",
    "(forall x. P(x) -> Q(x)) -> (P(a) -> Q(a))",
]

SAMPLE_BENCHMARKS = {
    "easy.txt": [
        "P(a) -> P(a)",
        "(P(a) && Q(a)) -> P(a)",
        "(P(a) && Q(a)) -> Q(a)",
        "P(a) -> (P(a) || Q(a))",
        "Q(a) -> (P(a) || Q(a))",
    ],
    "medium.txt": [
        "((P(a) -> Q(a)) && (Q(a) -> R(a))) -> (P(a) -> R(a))",
        "(forall x. P(x)) -> P(a)",
        "(forall x. P(x) -> Q(x)) -> (P(a) -> Q(a))",
        "((P(a) || Q(a)) && (P(a) -> R(a)) && (Q(a) -> R(a))) -> R(a)",
        "(exists x. P(x)) -> (exists x. P(x))",
    ],
    "hard.txt": [
        "(forall x. P(x) -> Q(x)) -> ((forall x. Q(x) -> R(x)) -> (P(a) -> R(a)))",
        "((forall x. P(x)) && (forall x. P(x) -> Q(x))) -> Q(a)",
        "(forall x. P(x) -> (Q(x) || R(x))) -> (P(a) -> (Q(a) || R(a)))",
        "((exists x. P(x)) && (forall x. P(x) -> Q(x))) -> (exists x. Q(x))",
        "(forall x. P(x) -> P(f(x))) -> (P(a) -> P(f(a)))",
    ],
}


def make_sample_benchmarks(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for filename, lines in SAMPLE_BENCHMARKS.items():
        (folder / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Created sample benchmark files in: {folder}")


def read_formula_lines(path: Path) -> List[str]:
    lines: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def run_one(prover_name: str, prover: BaselineProver, dataset: str, formula_id: int, formula_text: str) -> Dict[str, object]:
    try:
        formula = parse_formula(formula_text)
    except ParseError as exc:
        return {
            "dataset": dataset,
            "formula_id": formula_id,
            "prover": prover_name,
            "status": "parse_error",
            "nodes_expanded": 0,
            "elapsed_seconds": 0,
            "formula": formula_text,
            "error": str(exc),
        }

    try:
        result = prover.prove(formula)
        return {
            "dataset": dataset,
            "formula_id": formula_id,
            "prover": prover_name,
            "status": result.status,
            "nodes_expanded": result.nodes_expanded,
            "elapsed_seconds": round(result.elapsed, 6),
            "formula": formula_text,
            "error": "",
        }
    except RecursionError as exc:
        return {
            "dataset": dataset,
            "formula_id": formula_id,
            "prover": prover_name,
            "status": "recursion_error",
            "nodes_expanded": getattr(prover, "nodes_expanded", 0),
            "elapsed_seconds": round(time.perf_counter() - getattr(prover, "start_time", time.perf_counter()), 6),
            "formula": formula_text,
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "dataset": dataset,
            "formula_id": formula_id,
            "prover": prover_name,
            "status": "runtime_error",
            "nodes_expanded": getattr(prover, "nodes_expanded", 0),
            "elapsed_seconds": round(time.perf_counter() - getattr(prover, "start_time", time.perf_counter()), 6),
            "formula": formula_text,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_benchmarks(benchmark_folder: Path, results_csv: Path, max_nodes: int, max_seconds: float) -> None:
    rows: List[Dict[str, object]] = []
    files = sorted(benchmark_folder.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt benchmark files found in {benchmark_folder}")

    for file_path in files:
        formulae = read_formula_lines(file_path)
        for i, formula_text in enumerate(formulae, start=1):
            rows.append(run_one("baseline", BaselineProver(max_nodes=max_nodes, max_seconds=max_seconds), file_path.name, i, formula_text))
            rows.append(run_one("improved", ImprovedProver(max_nodes=max_nodes, max_seconds=max_seconds), file_path.name, i, formula_text))

    results_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "formula_id", "prover", "status", "nodes_expanded", "elapsed_seconds", "formula", "error"]
    with results_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote benchmark results to: {results_csv}")
    print_summary(rows)


def print_summary(rows: List[Dict[str, object]]) -> None:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["prover"]))
        grouped.setdefault(key, []).append(row)

    print("\nSummary")
    print("dataset, prover, proved, failed, timeout, parse_error, runtime_error, recursion_error, total_nodes, total_time")
    for (dataset, prover), items in sorted(grouped.items()):
        proved = sum(1 for r in items if r["status"] == "proved")
        failed = sum(1 for r in items if r["status"] == "failed")
        timeout = sum(1 for r in items if r["status"] == "timeout")
        parse_error = sum(1 for r in items if r["status"] == "parse_error")
        runtime_error = sum(1 for r in items if r["status"] == "runtime_error")
        recursion_error = sum(1 for r in items if r["status"] == "recursion_error")
        total_nodes = sum(int(r["nodes_expanded"]) for r in items)
        total_time = sum(float(r["elapsed_seconds"]) for r in items)
        print(f"{dataset}, {prover}, {proved}, {failed}, {timeout}, {parse_error}, {runtime_error}, {recursion_error}, {total_nodes}, {total_time:.6f}")


def run_examples() -> None:
    prover = BaselineProver(max_nodes=2000, max_seconds=2.0)
    for s in EXAMPLES:
        f = parse_formula(s)
        result = prover.prove(f)
        print(f"Formula: {s}")
        print(f"Parsed : {f}")
        print(f"Result : {result.status}, nodes={result.nodes_expanded}, time={result.elapsed:.4f}s")
        print("-")


def main() -> None:
    parser = argparse.ArgumentParser(description="3806ICT FOL sequent prover benchmark runner")
    parser.add_argument("--examples", action="store_true", help="run built-in example formulas")
    parser.add_argument("--make-benchmarks", action="store_true", help="create sample benchmark files")
    parser.add_argument("--benchmarks", default="benchmarks", help="benchmark folder path")
    parser.add_argument("--results", default="results/results.csv", help="output CSV path")
    parser.add_argument("--max-nodes", type=int, default=10000)
    parser.add_argument("--max-seconds", type=float, default=5.0)
    args = parser.parse_args()

    benchmark_folder = Path(args.benchmarks)
    results_csv = Path(args.results)

    if args.make_benchmarks:
        make_sample_benchmarks(benchmark_folder)
        return

    if args.examples:
        run_examples()
        return

    run_benchmarks(benchmark_folder, results_csv, args.max_nodes, args.max_seconds)


if __name__ == "__main__":
    main()
