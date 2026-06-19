from loaders import ImageRef
from pathlib import Path
import base64
import io
from PIL import Image
import pillow_avif  # noqa: F401  registers AVIF support in Pillow
from config import settings


# Dataset files have unreliable extensions (.jpg files are often AVIF/TIFF/WebP),
# and the API only accepts JPEG/PNG/GIF/WebP. Normalize everything to JPEG via
# Pillow so the real format and media_type always agree.
MAX_EDGE = 1536  # downscale the long edge to bound image tokens / cost


def encode_image(ref: ImageRef) -> dict | None:
    full_path = Path(settings.dataset_dir) / ref.path
    try:
        with Image.open(full_path) as im:
            im = im.convert("RGB")
            if max(im.size) > MAX_EDGE:
                im.thumbnail((MAX_EDGE, MAX_EDGE))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=90)
    except (FileNotFoundError, OSError, ValueError):
        return None

    b64 = base64.standard_b64encode(buf.getvalue()).decode()
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": b64,
        },
    }


def build_image_blocks(images: list[ImageRef]) -> tuple[list[dict], list[str]]:
    content_blocks = []
    missing_ids = []

    for ref in images:
        encoded = encode_image(ref)
        if encoded is None:
            missing_ids.append(ref.image_id)
            continue

        content_blocks.append({"type": "text", "text": f"Image {ref.image_id}:"})
        content_blocks.append(encoded)

    return content_blocks, missing_ids