from pathlib import Path
import pandas as pd
from dataclasses import dataclass


@dataclass
class ImageRef:
    image_id: str
    path: Path

    def __post_init__(self):
        self.path = Path(self.path)
        

@dataclass
class Claim:
    user_id: str
    image_paths_raw: str
    images: list[ImageRef]
    user_claim: str
    claim_object: str


def parse_image_refs(image_path: str) -> list[ImageRef]:
    return [
        ImageRef(image_id=Path(p).stem, path=p.strip())
        for p in image_path.split(';')
        if p.strip()
    ]


def load_claims(file_path: str) -> list[Claim]:
    df = pd.read_csv(file_path)
    return [
        Claim(
            user_id=row["user_id"],
            image_paths_raw=row["image_paths"],
            images=parse_image_refs(row["image_paths"]),
            user_claim=row["user_claim"],
            claim_object=row["claim_object"],
        )
        for _, row in df.iterrows()
    ]


@dataclass
class UserHistory:
    user_id: str
    past_claim_count: int
    accept_claim: str
    manual_review_claim: str
    rejected_claim: str
    last_90_days_claim_count: str
    history_flags: str
    history_summary: str


def load_user_history(path: str) -> list[UserHistory]:
    df = pd.read_csv(path)
    return [
        UserHistory(
            user_id=row["user_id"],
            past_claim_count=row["past_claim_count"],
            accept_claim=row["accept_claim"],
            manual_review_claim=row["manual_review_claim"],
            rejected_claim=row["rejected_claim"],
            last_90_days_claim_count=row["last_90_days_claim_count"],
            history_flags=row["history_flags"],
            history_summary=row["history_summary"],
        )
        for _, row in df.iterrows()
    ]


def index_user_history(user_id: str, path: str) -> list[UserHistory] | None:
    for h in load_user_history(path):
        if h.user_id == user_id:
            return h
    return None


@dataclass
class EvidenceReqs:
    requirement_id: str
    claim_object: str
    applies_to: str
    minimum_image_evidence: str


def load_evidence_requirements(path: str) -> list[EvidenceReqs]:
    df = pd.read_csv(path)
    return [
        EvidenceReqs(
            requirement_id=row["requirement_id"],
            claim_object=row["claim_object"],
            applies_to=row["applies_to"],
            minimum_image_evidence=row["minimum_image_evidence"],
        )
        for _, row in df.iterrows()
    ]


def index_evidence_requirements(path: str, claim_object: str, issue_family: str) -> list[EvidenceReqs]:
    requirements = load_evidence_requirements(path)
    return [
        req for req in requirements
        if req.claim_object in (claim_object, "all") or req.applies_to == issue_family
    ]