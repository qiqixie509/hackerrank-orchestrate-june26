"""Evaluate the system on the labeled sample set.

Runs `pipeline.process_claim` on the input columns of sample_claims.csv,
compares predictions to the gold labels, and prints per-field metrics.

Run from anywhere:
    uv run python code/evaluation/main.py            # run + score all 20
    uv run python code/evaluation/main.py --limit 5  # cost-controlled
    uv run python code/evaluation/main.py --predictions eval_predictions.csv
        # score an existing predictions file, no API calls
"""
import argparse
import csv
import os
import sys

# Make code/ importable and run from repo root so dataset paths resolve.
_THIS = os.path.abspath(__file__)
CODE_DIR = os.path.dirname(os.path.dirname(_THIS))
REPO_ROOT = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)
os.chdir(REPO_ROOT)

from config import settings           # noqa: E402
from loaders import load_claims       # noqa: E402
from pipeline import process_claim    # noqa: E402
from writer import write_output_csv   # noqa: E402

PRED_PATH = "eval_predictions.csv"

EXACT_FIELDS = [
    "claim_status",
    "issue_type",
    "object_part",
    "evidence_standard_met",
    "valid_image",
]
SET_FIELDS = ["risk_flags", "supporting_image_ids"]
SEVERITY_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}


# ---------- normalization ----------
def norm(value) -> str:
    return str(value).strip().lower()


def parse_set(value) -> frozenset:
    parts = [p.strip().lower() for p in str(value).split(";")]
    return frozenset(p for p in parts if p and p != "none")


# ---------- comparators ----------
def exact_accuracy(gold, pred, field) -> float:
    hits = sum(norm(g[field]) == norm(p[field]) for g, p in zip(gold, pred))
    return hits / len(gold) if gold else 0.0


def confusion_and_macro_f1(gold, pred, field):
    classes = sorted({norm(r[field]) for r in gold} | {norm(r[field]) for r in pred})
    confusion = {g: {p: 0 for p in classes} for g in classes}
    for g, p in zip(gold, pred):
        confusion[norm(g[field])][norm(p[field])] += 1

    f1s = []
    per_class = {}
    for c in classes:
        tp = sum(norm(g[field]) == c and norm(p[field]) == c
                 for g, p in zip(gold, pred))
        fp = sum(norm(p[field]) == c and norm(g[field]) != c
                 for g, p in zip(gold, pred))
        fn = sum(norm(g[field]) == c and norm(p[field]) != c
                 for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1}
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return confusion, per_class, macro_f1


def set_metrics(gold, pred, field):
    tp = fp = fn = 0
    exact = 0
    jaccards = []
    for g, p in zip(gold, pred):
        gs, ps = parse_set(g[field]), parse_set(p[field])
        tp += len(gs & ps)
        fp += len(ps - gs)
        fn += len(gs - ps)
        if gs == ps:
            exact += 1
        union = gs | ps
        jaccards.append(len(gs & ps) / len(union) if union else 1.0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "micro_precision": prec,
        "micro_recall": rec,
        "micro_f1": f1,
        "jaccard": sum(jaccards) / len(jaccards) if jaccards else 0.0,
        "exact_set_match": exact / len(gold) if gold else 0.0,
    }


def severity_metrics(gold, pred):
    n = len(gold)
    exact = sum(norm(g["severity"]) == norm(p["severity"])
                for g, p in zip(gold, pred))
    diffs = []
    for g, p in zip(gold, pred):
        gv, pv = norm(g["severity"]), norm(p["severity"])
        if gv in SEVERITY_ORDER and pv in SEVERITY_ORDER:
            diffs.append(abs(SEVERITY_ORDER[gv] - SEVERITY_ORDER[pv]))
    adjacent = sum(d <= 1 for d in diffs) / len(diffs) if diffs else 0.0
    mae = sum(diffs) / len(diffs) if diffs else 0.0
    return {
        "exact": exact / n if n else 0.0,
        "within_one": adjacent,
        "mae_ordinal": mae,
    }


# ---------- io ----------
def load_rows(path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_predictions(limit) -> list:
    claims = load_claims(settings.sample_csv)
    if limit is not None:
        claims = claims[:limit]
    rows = []
    for i, claim in enumerate(claims, 1):
        row, _ = process_claim(claim)
        rows.append(row)
        print(f"[{i}/{len(claims)}] {claim.user_id} -> {row['claim_status']}")
    write_output_csv(rows, PRED_PATH)
    print(f"Saved predictions to {PRED_PATH}\n")
    return rows


# ---------- report ----------
def report(gold, pred) -> None:
    n = len(gold)
    print(f"\n===== EVALUATION ({n} rows) =====")

    confusion, per_class, macro_f1 = confusion_and_macro_f1(
        gold, pred, "claim_status")
    acc = exact_accuracy(gold, pred, "claim_status")
    print("\n[claim_status]  (primary)")
    print(f"  accuracy={acc:.3f}  macro_f1={macro_f1:.3f}")
    for c, m in per_class.items():
        print(f"    {c:25s} P={m['precision']:.2f} "
              f"R={m['recall']:.2f} F1={m['f1']:.2f}")
    print("  confusion (gold rows -> pred cols):")
    cols = list(confusion.keys())
    print("    " + " ".join(f"{c[:6]:>7}" for c in cols))
    for g in cols:
        line = " ".join(f"{confusion[g][p]:>7}" for p in cols)
        print(f"    {g[:6]:>6}{line}")

    print("\n[categorical accuracy]")
    for field in EXACT_FIELDS:
        print(f"  {field:24s} acc={exact_accuracy(gold, pred, field):.3f}")

    print("\n[severity] (ordinal)")
    for k, v in severity_metrics(gold, pred).items():
        print(f"  {k:24s} {v:.3f}")

    print("\n[set fields]")
    for field in SET_FIELDS:
        m = set_metrics(gold, pred, field)
        print(f"  {field}")
        for k, v in m.items():
            print(f"    {k:18s} {v:.3f}")

    print("\n[mismatched rows: claim_status]")
    for g, p in zip(gold, pred):
        if norm(g["claim_status"]) != norm(p["claim_status"]):
            print(f"  {g['user_id']}: gold={g['claim_status']} "
                  f"pred={p['claim_status']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on the sample set.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--predictions", default=None,
        help="Score an existing predictions CSV (no API calls).")
    args = parser.parse_args()

    if args.predictions:
        pred = load_rows(args.predictions)
    else:
        pred = run_predictions(args.limit)

    gold = load_rows(settings.sample_csv)[: len(pred)]
    report(gold, pred)


if __name__ == "__main__":
    main()