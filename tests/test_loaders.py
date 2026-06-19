import textwrap
import pytest
from pathlib import Path
from loaders import load_claims, parse_image_refs


def test_parse_image_refs():
    images = parse_image_refs("images/test/case_001/img_1.jpg;images/test/case_001/img_2.jpg;images/test/case_001/img_3.jpg")
    assert len(images) == 3
    assert images[0].image_id == "img_1"
    assert images[0].path == "images/test/case_001/img_1.jpg"
    assert images[1].image_id == "img_2"
    assert images[1].path == "images/test/case_001/img_2.jpg"
    assert images[2].image_id == "img_3"
    assert images[2].path == "images/test/case_001/img_3.jpg"


def test_load_claims():
    file_path = "dataset/claims.csv"
    claims = load_claims(file_path)
    assert len(claims) == 44
    assert claims[0].user_id == "user_002"