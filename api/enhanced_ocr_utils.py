"""
Helpers that (1) clean / enhance a cropped plate before OCR,
(2) pick the best OCR line, and (3) fix state-specific quirks.

Requires: easyocr, numpy, opencv-python, Pillow.
"""

import re
import cv2
import numpy as np
from PIL import Image, ImageEnhance

# ----------------------------  OCR PROCESSOR  ---------------------------- #

class EnhancedOCRProcessor:
    """
    Wrapper around EasyOCR that:
      • boosts contrast / sharpness
      • upsamples tiny crops
      • filters EasyOCR results so garden-variety text
        like 'Garden State' or dealership frames is ignored
    """
    def __init__(self, easy_reader, min_conf=0.30):
        self.reader = easy_reader
        self.min_conf = min_conf

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Upsample if tiny, then bump contrast & sharpness."""
        if crop is None or crop.size == 0:
            return crop

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        # --- upscale if width < 200 px ----------------------------------- #
        if pil.width < 200:
            scale = 200 / pil.width
            pil = pil.resize(
                (int(pil.width * scale), int(pil.height * scale)),
                Image.Resampling.LANCZOS
            )

        # --- basic enhancements ----------------------------------------- #
        pil = ImageEnhance.Contrast(pil).enhance(2.0)
        pil = ImageEnhance.Sharpness(pil).enhance(2.0)

        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    def _filter_ocr_boxes(self, ocr_raw):
        """
        Keep only boxes that are (i) reasonably confident,
        (ii) mostly alphanumeric, (iii) about the right aspect ratio.
        """
        cleaned = []
        for txt, bbox, conf in ocr_raw:
            if conf < self.min_conf:
                continue

            # Strip punctuation / spaces for ratio check
            txt_flat = re.sub(r'[^A-Za-z0-9]', '', txt)
            if not txt_flat:
                continue

            h = bbox[1][1] - bbox[0][1]
            w = bbox[1][0] - bbox[0][0]
            if h == 0:
                continue
            aspect = w / h

            # US plates are roughly 2-4× wider than tall.
            if not 1.5 <= aspect <= 5.0:
                continue

            # At least 4 characters & majority alnum
            if len(txt_flat) < 4 or sum(ch.isalnum() for ch in txt_flat) / len(txt_flat) < 0.8:
                continue

            cleaned.append((txt_flat.upper(), conf))

        return cleaned

    # ---------------------------  PUBLIC API  ---------------------------- #

    def process_plate(self, crop: np.ndarray):
        """
        Returns (best_text:str|None, confidence:float|0.0)
        """
        if crop is None or crop.size == 0:
            return None, 0.0

        image = self._preprocess(crop)
        ocr_raw = self.reader.readtext(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            width_ths=0.7,
            height_ths=0.7
        )
        candidates = self._filter_ocr_boxes(ocr_raw)
        if not candidates:
            return None, 0.0

        # Pick highest confidence – break ties by longer string
        best = sorted(candidates, key=lambda x: (x[1], len(x[0])), reverse=True)[0]
        return best

# -----------------------  STATE-AWARE CORRECTIONS  ----------------------- #

_state_rules = {
    # Template:
    # "STATE_CODE": [(regex_pattern, replacement), ...]
    "NJ": [
        # EasyOCR often sees '0' (zero) as 'O' in NJ fonts
        (r"O", "0"),
    ],
    "NY": [
        (r"O", "0"),
    ],
    "CA": [
        (r"[^\dA-Z]", ""),   # strip extra punctuation / spaces
    ],
    # Extend as needed …
}

def apply_ocr_corrections_for_state(text: str, state_code: str | None) -> str:
    """Return corrected text if a rule matches, else original text."""
    if not text or not state_code:
        return text

    rules = _state_rules.get(state_code.upper(), [])
    for pattern, repl in rules:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text.upper()
