"""Developer tool: trace one Ask-the-Paper question up to (not into) generation.

Answers the question "why did this abstain?" by printing every intermediate the
deterministic pipeline produced: the retrieval pool before truncation, which
passages reached the answerability window, the term coverage and gate inputs
that decided the verdict, and the evidence spans that would have been cited.

Read-only. It imports the production functions and calls them in the production
order taken from grounded_answer.answer_question, stopping before
generate_grounded_answer, so no answer model is ever called. Nothing here
re-implements retrieval or answerability: the status and reason printed are the
ones production produced.

    python ask_diagnostics.py "/path/to/paper.pdf" "What did the ETAU group receive?"
    python ask_diagnostics.py paper.pdf "..." --json > trace.json
    python ask_diagnostics.py paper.pdf "..." --top-k 10     # investigation only

Requires Ollama with the embedding model pulled (`ollama pull nomic-embed-text`).
The generation model is never contacted.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from answerability import (
    asks_for_specific_attribute,
    attribute_present,
    content_terms,
    detect_question_types,
    evidence_text,
    focus_terms,
    summarise,
    term_coverage,
    verify_answerability,
)
from config import SELECTION_POOL_SIZE
from embeddings import embed_chunks, rank_with_diagnostics
from evidence_ranking import (
    CANDIDATE_MULTIPLIER,
    MIN_CANDIDATE_POOL,
    carries_question_evidence,
    retrieve_evidence,
)
from grounded_answer import MAX_EVIDENCE_CHUNKS, select_evidence
from pdf import extract_document
from retrieval import build_retrieval_chunks
from schemas import AskRequest, PageInput

# The reason strings verify_answerability emits, mapped to the rule that
# produced them. Read off production's own output rather than re-deciding:
# duplicating the branch conditions here is exactly how a diagnostic starts
# disagreeing with the thing it is meant to explain.
RULE_BY_REASON = (
    ("No eligible passages", "retrieval returned nothing"),
    ("topically related but contain no", "question-type veto (gate C)"),
    ("describe the study but never report", "specific-attribute gate (gate C2)"),
    ("no retrieved passage shares enough", "term-coverage gate (gate B)"),
    ("too short to verify", "short-question fallback (< MIN_QUESTION_TERMS)"),
    ("close to noise", "top score < MIN_USEFUL_TOP_SCORE"),
    ("A single passage", "only one supporting passage (< MIN_SUPPORTING_CHUNKS)"),
    ("Relevant evidence found", "supported"),
)


def responsible_rule(reason: str) -> str:
    for fragment, rule in RULE_BY_REASON:
        if fragment in reason:
            return rule
    return "unrecognised reason string"


def load_pages(path: str) -> List[PageInput]:
    data = Path(path).read_bytes()
    document = extract_document(data, Path(path).name)
    return [PageInput(**page) for page in document["pages"]]


def snippet(text: str, limit: int = 160) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


async def trace(
    question: str, filename: str, pages: Sequence[PageInput], top_k: int
) -> Dict[str, Any]:
    """Reproduce answer_question's sequence, recording every intermediate."""
    chunks = build_retrieval_chunks(pages, filename)
    embedded = await embed_chunks(chunks)

    out: Dict[str, Any] = {
        "question": question,
        "filename": filename,
        "top_k": top_k,
        "chunks_built": len(chunks),
        "chunks_embedded": len(embedded),
    }

    if not embedded:
        out["error"] = "The paper produced no passages that could be searched."
        return out

    # -- question analysis (the same helpers the verifier uses) --------------
    detected = detect_question_types(question)
    question_terms = content_terms(question)
    out["question_terms"] = sorted(question_terms)
    out["detected_types"] = [
        {"name": qt.name, "label": qt.label, "precise": qt.precise}
        for qt, _ in detected
    ]
    out["asks_specific_attribute"] = asks_for_specific_attribute(question)
    out["focus_terms"] = sorted(focus_terms(question))

    # -- the candidate pool, before retrieve_evidence truncates to top_k -----
    pool_size = max(MIN_CANDIDATE_POOL, top_k * CANDIDATE_MULTIPLIER)
    pool = await rank_with_diagnostics(question, embedded, top_k=pool_size)
    out["pool_size"] = pool_size
    out["pool_diagnostics"] = pool.diagnostics.model_dump()
    out["candidates"] = [
        {
            "rank": rank,
            "chunk_id": chunk.chunk_id,
            "page": chunk.page_number,
            "section": chunk.section or "(unlabelled)",
            "cosine": round(chunk.score, 4),
            "cue_bearing": carries_question_evidence(question, chunk),
            "text": snippet(chunk.text),
        }
        for rank, chunk in enumerate(pool.results)
    ]

    # -- production calls, in production order -------------------------------
    ranking = await retrieve_evidence(question, embedded, top_k=top_k)
    report = verify_answerability(question, ranking)

    window_ids = [chunk.chunk_id for chunk in ranking.results]
    out["window_chunk_ids"] = window_ids
    combined = "\n".join(evidence_text(chunk) for chunk in ranking.results)
    evidence_terms = content_terms(combined)

    out["window"] = []
    for rank, chunk in enumerate(ranking.results):
        coverage, matched = term_coverage(question_terms, evidence_text(chunk))
        out["window"].append(
            {
                "rank": rank,
                "chunk_id": chunk.chunk_id,
                "page": chunk.page_number,
                "section": chunk.section or "(unlabelled)",
                "cosine": round(chunk.score, 4),
                "term_coverage": round(coverage, 3),
                "matched_terms": sorted(matched),
                "cue_bearing": carries_question_evidence(question, chunk),
                "supporting": chunk.chunk_id in report.supporting_chunk_ids,
            }
        )

    # Gate inputs, observed against the same combined text the verifier used.
    out["type_gate"] = [
        {
            "name": qt.name,
            "label": qt.label,
            "evidence_present": qt.evidence_present(combined, match),
        }
        for qt, match in detected
    ]
    out["attribute_gate"] = {
        "applies": out["asks_specific_attribute"],
        "terms_present": {
            term: attribute_present(term, evidence_terms, combined)
            for term in sorted(focus_terms(question))
        },
    }

    out["answerability"] = summarise(report)
    out["responsible_rule"] = responsible_rule(report.reason)

    # -- selection stage, only when production would run it ------------------
    answerable = report.is_answerable
    candidates = ranking.results
    if answerable and len(embedded) > top_k:
        wider = await retrieve_evidence(
            question, embedded, top_k=max(top_k, SELECTION_POOL_SIZE)
        )
        candidates = wider.results
    out["selection_pool"] = [
        {
            "rank": rank,
            "chunk_id": chunk.chunk_id,
            "page": chunk.page_number,
            "section": chunk.section or "(unlabelled)",
        }
        for rank, chunk in enumerate(candidates)
    ]

    if answerable:
        selected = select_evidence(report, candidates, question, MAX_EVIDENCE_CHUNKS)
        out["selected_evidence"] = [
            {
                "chunk_id": item.chunk_id,
                "page": item.page_number,
                "section": item.section or "(unlabelled)",
                "score": round(item.score, 4),
                "span": item.span,
            }
            for item in selected
        ]
    else:
        out["selected_evidence"] = []
        out["note"] = "NOT_SUPPORTED: production abstains here without generating."

    return out


def render(out: Dict[str, Any]) -> str:
    lines: List[str] = []
    add = lines.append

    add("=" * 78)
    add("QUESTION: {}".format(out["question"]))
    add("PAPER   : {}  ({} chunks built, {} embedded)".format(
        out["filename"], out.get("chunks_built", 0), out.get("chunks_embedded", 0)))
    add("=" * 78)
    if "error" in out:
        add("ERROR: " + out["error"])
        return "\n".join(lines)

    add("")
    add("-- 1. QUESTION ANALYSIS " + "-" * 54)
    add("  content terms      : {}".format(", ".join(out["question_terms"]) or "(none)"))
    types = out["detected_types"]
    add("  detected types     : {}".format(
        ", ".join("{}{}".format(t["name"], "*" if t["precise"] else "") for t in types)
        or "(none)"))
    add("                       (* = precise; a precise type can veto)")
    add("  attribute question : {}".format(out["asks_specific_attribute"]))
    if out["focus_terms"]:
        add("  focus terms        : {}".format(", ".join(out["focus_terms"])))

    add("")
    add("-- 2. RETRIEVAL POOL (top {}, before truncation to top_k={}) {}".format(
        out["pool_size"], out["top_k"], "-" * 12))
    diag = out["pool_diagnostics"]
    add("  considered={considered} excluded={excluded} returned={returned}".format(**diag))
    if diag.get("excluded_by_reason"):
        add("  excluded_by_reason: {}".format(diag["excluded_by_reason"]))
    add("  top={} second={} gap={} mean={}".format(
        diag.get("top_score"), diag.get("second_score"),
        diag.get("score_gap"), diag.get("mean_top_k")))
    add("")
    add("  {:<4} {:<7} {:<5} {:<26} {:<4} {}".format(
        "rank", "cosine", "page", "section", "cue", "chunk_id"))
    window = set(out["window_chunk_ids"])
    for cand in out["candidates"]:
        marker = ">>" if cand["chunk_id"] in window else "  "
        add("{}{:<4} {:<7} {:<5} {:<26} {:<4} {}".format(
            marker, cand["rank"], cand["cosine"], cand["page"],
            cand["section"][:26], "yes" if cand["cue_bearing"] else "-",
            cand["chunk_id"]))
        add("      {}".format(cand["text"]))
    add("")
    add("  ('>>' marks the {} passages that entered the answerability window)".format(
        len(window)))

    add("")
    add("-- 3. ANSWERABILITY WINDOW " + "-" * 51)
    add("  {:<4} {:<7} {:<9} {:<5} {:<22} {}".format(
        "rank", "cosine", "coverage", "page", "section", "supporting"))
    for item in out["window"]:
        add("  {:<4} {:<7} {:<9} {:<5} {:<22} {}".format(
            item["rank"], item["cosine"], item["term_coverage"], item["page"],
            item["section"][:22], "yes" if item["supporting"] else "no"))
        add("       matched: {}".format(", ".join(item["matched_terms"]) or "(none)"))

    add("")
    add("-- 4. GATES " + "-" * 66)
    if out["type_gate"]:
        for gate in out["type_gate"]:
            add("  type '{}' ({}): evidence_present={}".format(
                gate["name"], gate["label"], gate["evidence_present"]))
    else:
        add("  no question types detected (no veto possible)")
    attr = out["attribute_gate"]
    add("  specific-attribute gate applies: {}".format(attr["applies"]))
    for term, present in attr["terms_present"].items():
        add("     '{}' present in evidence: {}".format(term, present))

    add("")
    add("-- 5. VERDICT " + "-" * 64)
    verdict = out["answerability"]
    add("  status  : {}".format(verdict["status"]))
    add("  rule    : {}".format(out["responsible_rule"]))
    add("  reason  : {}".format(verdict["reason"]))
    add("  pages   : {}".format(verdict.get("pages")))
    if verdict.get("missing_evidence"):
        add("  missing : {}".format(verdict["missing_evidence"]))

    add("")
    add("-- 6. EVIDENCE SELECTION " + "-" * 53)
    if out["selected_evidence"]:
        add("  selection pool: {} passages".format(len(out["selection_pool"])))
        for item in out["selected_evidence"]:
            add("  [{}] p{} {} score={}".format(
                item["chunk_id"], item["page"],
                item["section"], item["score"]))
            add("      {}".format(snippet(item["span"], 300)))
    else:
        add("  {}".format(out.get("note", "no evidence selected")))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ask_diagnostics.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pdf", help="path to the paper")
    parser.add_argument("question", help="the question to trace")
    parser.add_argument(
        "--top-k",
        type=int,
        default=AskRequest.model_fields["top_k"].default,
        help="answerability window size; defaults to the /ask default. "
             "Changing it here investigates only - production is untouched.",
    )
    parser.add_argument("--json", action="store_true", help="emit the raw trace")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.pdf)
    if not path.is_file():
        print("No such file: {}".format(path), file=sys.stderr)
        return 2

    pages = load_pages(str(path))
    out = asyncio.run(trace(args.question, path.name, pages, args.top_k))
    print(json.dumps(out, indent=2) if args.json else render(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
