import csv
from loaders import Claim
from schema import ClaimPrediction


# Exact column order required by problem_statement.md
OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _join(values: list) -> str:
    """Join a list of enum/str values into a semicolon string, dropping
    'none'/empty entries. Returns 'none' if nothing is left."""
    items = [getattr(v, "value", v) for v in values]
    items = [str(v).strip() for v in items if v and str(v).strip().lower() != "none"]
    return ";".join(items) if items else "none"


def to_row(claim: Claim, pred: ClaimPrediction) -> dict:
    """Merge the verbatim input fields with the model prediction into one
    output row with values only (enums -> value, bools -> true/false,
    lists -> semicolon strings)."""
    return {
        "user_id": claim.user_id,
        "image_paths": claim.image_paths_raw,
        "user_claim": claim.user_claim,
        "claim_object": claim.claim_object,
        "evidence_standard_met": _bool_str(pred.evidence_standard_met),
        "evidence_standard_met_reason": pred.evidence_standard_met_reason,
        "risk_flags": _join(pred.risk_flags),
        "issue_type": pred.issue_type.value,
        "object_part": pred.object_part.value,
        "claim_status": pred.claim_status.value,
        "claim_status_justification": pred.claim_status_justification,
        "supporting_image_ids": _join(pred.supporting_image_ids),
        "valid_image": _bool_str(pred.valid_image),
        "severity": pred.severity.value,
    }


def write_output_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)