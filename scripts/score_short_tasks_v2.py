#!/usr/bin/env python3
"""Versioned bounded content grading of frozen qbh tasks; never modifies v1.

Unknown prose is unresolved (not automatically a logic error). No substring
matching, model judges, answer-specific output allowlists or inference required.
"""
import argparse
import ast
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from eval_exp0218 import grade as strict_grade

VERSION = "qbh-content-v2"
NUMBER = r"[+-]?\d+(?:\.\d+)?"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def surface(text):
    text = unicodedata.normalize("NFKC", text).strip()
    fence = re.fullmatch(r"```(?:json|text)?\s*\n(.*?)\n```", text, re.S | re.I)
    if fence:
        text = fence[1].strip()
    # Only remove balanced presentation wrappers around the entire answer.
    for _ in range(4):
        for left, right in [("**", "**"), ("`", "`"), ('"', '"'),
                            ("'", "'"), ("“", "”"), ("‘", "’")]:
            if (text.startswith(left) and text.endswith(right) and len(text) > len(left)+len(right)
                    and left not in text[len(left):-len(right)]):
                text = text[len(left):-len(right)].strip()
                break
        else:
            break
    return text


def typed_equal(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(typed_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(typed_equal(x, y) for x, y in zip(a, b))
    return a == b


def unique_object(pairs):
    obj = {}
    for k, v in pairs:
        if k in obj:
            raise ValueError("duplicate JSON key")
        obj[k] = v
    return obj


def arithmetic(expression):
    """Exact bounded arithmetic, no eval/calls/names/exponents."""
    if len(expression) > 128 or not re.fullmatch(r"[\d.\s+*/()\-]+", expression):
        raise ValueError("unsupported expression")
    root = ast.parse(expression.strip(), mode="eval")
    if len(list(ast.walk(root))) > 40:
        raise ValueError("expression too complex")

    def walk(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = Fraction(str(node.value))
            if abs(value) > 10**12:
                raise ValueError("constant too large")
            return value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return walk(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a, b = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Add): return a+b
            if isinstance(node.op, ast.Sub): return a-b
            if isinstance(node.op, ast.Mult): return a*b
            return a/b
        raise ValueError("unsupported arithmetic")
    return walk(root.body)


def result(strict, correct, answer, reason):
    category = ("strict_correct" if strict else "format_only") if correct else (
        "answer_error" if correct is False else "unresolved")
    return dict(scorer_version=VERSION, strict_correct=strict,
                content_correct=correct, classification=category,
                extracted_answer=answer, reason=reason)


def grade_content(sample, raw):
    strict = strict_grade(sample, raw)
    text = surface(raw)
    scorer = sample["scorer"]
    if scorer == "json":
        try:
            # Do not NFKC-normalize content inside JSON string literals.
            value = json.loads(surface_json(raw), object_pairs_hook=unique_object,
                               parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
        except (ValueError, TypeError):
            return result(strict, None, None, "malformed_or_unsupported_JSON; manual review required")
        ok = typed_equal(value, sample["answer"])
        return result(strict, ok, value, "typed_JSON_equality" if ok else "wrong_JSON_keys_values_or_types")

    # Strip sentence-ending punctuation; never strip signs or internal punctuation.
    text = text.rstrip("。.!！").strip()
    text = re.sub(r"^(?:(?:the\s+)?(?:answer|result)\s*(?:is\s+|:\s*)|(?:答案|结果)(?:是|为|[:：])\s*)", "", text, flags=re.I)
    text = surface(text)
    if scorer == "numeric":
        if re.fullmatch(NUMBER, text):
            ok = Fraction(text) == Fraction(str(sample["answer"]))
            return result(strict, ok, text, "numeric_value" if ok else "wrong_numeric_answer")
        expr = text.replace("×", "*").replace("÷", "/").replace("−", "-")
        for word, op in [("divided by", "/"), ("plus", "+"), ("minus", "-"), ("times", "*")]:
            expr = re.sub(r"\b"+word+r"\b", op, expr, flags=re.I)
        expr = re.sub(r"\b(?:is|equals)\b", "=", expr, flags=re.I)
        if expr.count("=") == 1:
            lhs, rhs = (x.strip() for x in expr.split("="))
            if re.fullmatch(NUMBER, rhs):
                try:
                    computed = arithmetic(lhs)
                except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
                    return result(strict, None, None, "unsupported_or_invalid_arithmetic")
                if computed != Fraction(rhs):
                    return result(strict, False, rhs, "contradictory_arithmetic_equation")
                ok = Fraction(rhs) == Fraction(str(sample["answer"]))
                return result(strict, ok, rhs, "verified_equation" if ok else "wrong_equation_result")
        return result(strict, None, None, "no_unique_supported_numeric_assertion; manual review required")

    if scorer != "text":
        raise ValueError("unknown task scorer: "+scorer)
    answers = sample["answer"]
    # Repeated copies of one short literal carry the same content.
    lines = [surface(t).rstrip("。.!！").strip() for t in text.splitlines() if t.strip()]
    if len(lines) > 1 and len(set(lines)) == 1:
        text = lines[0]
    if text in answers:
        return result(strict, True, text, "literal_content_equality")
    prompt = sample.get("prompt", "")
    patterns = []
    if "谁负责记录" in prompt:
        patterns = [r"(.{1,12})负责记录", r"负责记录的是(.{1,12})"]
    elif "Who takes notes" in prompt:
        patterns = [r"([A-Za-z][A-Za-z -]{0,40}) takes notes"]
    elif "寄件人叫什么" in prompt:
        patterns = [r"寄件人(?:是|叫)(.{1,12})"]
    elif "sender's name" in prompt:
        patterns = [r"(?:The |the )?sender(?:'s name)? is ([A-Za-z][A-Za-z -]{0,40})"]
    elif "会议最终" in prompt:
        patterns = [r"(?:根据题目信息[,，]\s*)?会议最终(?:在|改到)(.{1,8}?)(?:举行)?"]
    elif "On which day" in prompt:
        patterns = [r"(?:The |the )?meeting (?:is|will take place) (?:on )?([A-Za-z]+)"]
    # Case conversion remains case-sensitive: CAT != cat != CATE.
    elif "uppercase" in prompt or "转换成大写" in prompt:
        patterns = [r'the word cat converted to uppercase is "([A-Za-z]+)"',
                    r'\*\*cat\*\*\s*→\s*\*\*([A-Za-z]+)\*\*']
    for pattern in patterns:
        match = re.fullmatch(pattern, text)
        if match:
            answer = match[1].strip()
            ok = answer in answers
            return result(strict, ok, answer, "explicit_answer_statement" if ok else "wrong_asserted_text_answer")
    if re.fullmatch(r"[A-Za-z]+", text):
        return result(strict, False, text, "wrong_literal_answer")
    return result(strict, None, None, "unsupported_text_assertion; manual review required")


def surface_json(raw):
    text = raw.strip()
    match = re.fullmatch(r"```(?:json)?[ \t]*\n(.*?)\n```", text, re.S | re.I)
    return match[1].strip() if match else text


def rescore(dataset, snapshot):
    tasks = {s["id"]: s for s in dataset["samples"] if s["kind"] == "task" and s["split"] == "full"}
    output = {}
    for variant, samples in snapshot["samples"].items():
        if not isinstance(samples, list):
            raise ValueError("expected quality_summary variant-to-samples mapping")
        rows = [s for s in samples if s["kind"] == "task"]
        if len(rows) != len(tasks) or {s["id"] for s in rows} != set(tasks):
            raise ValueError("missing/duplicate/extra task IDs")
        graded = []
        for row in rows:
            sample = tasks[row["id"]]
            grade = grade_content(sample, row["text"])
            if grade["strict_correct"] != row["correct"]:
                raise ValueError(f"historical strict grade mismatch: {variant}/{row['id']}")
            graded.append(dict(id=row["id"], language=sample["language"],
                               category=sample["category"], prompt=sample["prompt"],
                               expected=sample["answer"], text=row["text"],
                               token_ids=row["token_ids"], **grade))
        counts = Counter(g["classification"] for g in graded)
        output[variant] = dict(strict_correct=sum(g["strict_correct"] for g in graded),
                               content_correct=sum(g["content_correct"] is True for g in graded),
                               total=len(graded), counts=dict(counts), samples=graded)
    return dict(scorer_version=VERSION, retrospective=True, variants=output,
                original_quality_summary=snapshot["summary"],
                nll_and_generated_tokens_changed=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--snapshot", type=Path, required=True, help="existing quality_summary.json")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    data = rescore(json.loads(args.dataset.read_text()), json.loads(args.snapshot.read_text()))
    data["provenance"] = {"dataset": str(args.dataset.resolve()), "dataset_sha256": sha(args.dataset),
                          "snapshot": str(args.snapshot.resolve()), "snapshot_sha256": sha(args.snapshot),
                          "scorer_sha256": sha(__file__),
                          "strict_scorer_sha256": sha(Path(__file__).with_name("eval_exp0218.py"))}
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2)+"\n")
    for name, value in data["variants"].items():
        print(name, {k: v for k, v in value.items() if k != "samples"})


if __name__ == "__main__":
    main()
