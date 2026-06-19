from config import settings
from loaders import Claim, index_user_history, index_evidence_requirements
from images import build_image_blocks
from prompt import build_system_prompt, build_user_message
from model_client import predict
from schema import (
    ClaimPrediction,
    ClaimStatus,
    IssueType,
    ObjectPart,
    Severity,
    RiskFlag,
)
from writer import to_row

# Built once and reused across claims so the cached system-prompt prefix stays
# byte-identical (prompt caching).
SYSTEM_PROMPT = build_system_prompt()


def _history_summary(user_id: str) -> str:
    h = index_user_history(user_id, settings.user_history_csv)
    if h is None:
        return "No prior history is available for this user."
    return (
        f"{h.history_summary} "
        f"(past_claims={h.past_claim_count}, last_90_days={h.last_90_days_claim_count}, "
        f"flags={h.history_flags})"
    )


def _evidence_text(claim_object: str) -> str:
    # issue_family is unknown before the model runs, so gather every rule that
    # applies to this object (plus the object-agnostic "all" rules) and let the
    # model use them.
    reqs = index_evidence_requirements(settings.evidence_csv, claim_object, issue_family="")
    if not reqs:
        return "No specific evidence requirement is defined; use general judgment."
    return "\n".join(f"- {r.minimum_image_evidence}" for r in reqs)


def _fallback_prediction(reason: str) -> ClaimPrediction:
    """Used when no image is usable or the model call fails, so every input row
    still produces a valid output row."""
    return ClaimPrediction(
        evidence_standard_met=False,
        evidence_standard_met_reason=reason,
        risk_flags=[RiskFlag.manual_review_required],
        issue_type=IssueType.unknown,
        object_part=ObjectPart.unknown,
        claim_status=ClaimStatus.not_enough_information,
        claim_status_justification=reason,
        supporting_image_ids=[],
        valid_image=False,
        severity=Severity.unknown,
    )


def process_claim(claim: Claim) -> tuple[dict, dict]:
    """Run one claim end to end. Returns (output_row, usage)."""
    image_blocks, missing_ids = build_image_blocks(claim.images)

    # No usable image -> don't spend a model call.
    if not image_blocks:
        reason = f"No usable image could be loaded (missing/unreadable: {missing_ids})."
        return to_row(claim, _fallback_prediction(reason)), {}

    user_content = build_user_message(
        claim,
        history_summary=_history_summary(claim.user_id),
        evidence_text=_evidence_text(claim.claim_object),
        image_blocks=image_blocks,
    )

    try:
        pred, usage = predict(SYSTEM_PROMPT, user_content)
    except Exception as exc:  # keep one-row-per-input guarantee on API errors
        reason = f"Prediction failed: {type(exc).__name__}: {exc}"
        return to_row(claim, _fallback_prediction(reason)), {}

    return to_row(claim, pred), usage
