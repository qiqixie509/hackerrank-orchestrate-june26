from enum import Enum
from pydantic import BaseModel


class ClaimStatus(str, Enum):
    supported = "supported"
    contradicted = "contradicted"
    not_enough_information = "not_enough_information"


class IssueType(str, Enum):
    dent = "dent"
    scratch = "scratch"
    crack = "crack"
    glass_shatter = "glass_shatter"
    broken_part = "broken_part"
    missing_part = "missing_part"
    torn_packaging = "torn_packaging"
    crushed_packaging = "crushed_packaging"
    water_damage = "water_damage"
    stain = "stain"
    none = "none"
    unknown = "unknown"


class ObjectPart(str, Enum):
    # car
    front_bumper = "front_bumper"
    rear_bumper = "rear_bumper"
    door = "door"
    hood = "hood"
    windshield = "windshield"
    side_mirror = "side_mirror"
    headlight = "headlight"
    taillight = "taillight"
    fender = "fender"
    quarter_panel = "quarter_panel"
    body = "body"
    # laptop
    screen = "screen"
    keyboard = "keyboard"
    trackpad = "trackpad"
    hinge = "hinge"
    lid = "lid"
    corner = "corner"
    port = "port"
    base = "base"
    # package
    box = "box"
    package_corner = "package_corner"
    package_side = "package_side"
    seal = "seal"
    label = "label"
    contents = "contents"
    item = "item"
    # fallback
    unknown = "unknown"


class Severity(str, Enum):
    none = "none"
    low = "low"
    medium = "medium"
    high = "high"
    unknown = "unknown"


class RiskFlag(str, Enum):
    none = "none"
    blurry_image = "blurry_image"
    cropped_or_obstructed = "cropped_or_obstructed"
    low_light_or_glare = "low_light_or_glare"
    wrong_angle = "wrong_angle"
    wrong_object = "wrong_object"
    wrong_object_part = "wrong_object_part"
    damage_not_visible = "damage_not_visible"
    claim_mismatch = "claim_mismatch"
    possible_manipulation = "possible_manipulation"
    non_original_image = "non_original_image"
    text_instruction_present = "text_instruction_present"
    user_history_risk = "user_history_risk"
    manual_review_required = "manual_review_required"


class ClaimPrediction(BaseModel):
    """Only the model-predicted fields. The 4 input fields (user_id,
    image_paths, user_claim, claim_object) are added verbatim by the writer."""

    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: list[RiskFlag]
    issue_type: IssueType
    object_part: ObjectPart
    claim_status: ClaimStatus
    claim_status_justification: str
    supporting_image_ids: list[str]
    valid_image: bool
    severity: Severity
