import json
import math
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from scipy import ndimage


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p for p in parts]


def list_episode_dirs(root: Path) -> List[Path]:
    root = Path(root)
    return sorted(
        [p for p in root.iterdir() if p.is_dir() and (p / "images").is_dir()],
        key=natural_key,
    )


def list_images(episode_dir: Path) -> List[Path]:
    image_dir = Path(episode_dir) / "images"
    return sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS],
        key=natural_key,
    )


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def save_rgb(array: np.ndarray, path: Path, quality: int = 95):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")
    img.save(path, quality=quality)


def load_annotations(path: Path) -> Dict:
    path = Path(path)
    if not path.exists():
        return {"version": 1, "episodes": {}}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("version", 1)
    data.setdefault("episodes", {})
    return data


def save_annotations(path: Path, data: Dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp.replace(path)


def parse_bbox(value: str) -> Tuple[int, int, int, int]:
    parts = [int(float(p.strip())) for p in value.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be x0,y0,x1,y1")
    return normalize_bbox(parts)


def normalize_bbox(bbox: Sequence[int]) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def clip_bbox(
    bbox: Sequence[int], width: int, height: int, padding: int = 0
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = normalize_bbox(bbox)
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(width, x1 + padding)
    y1 = min(height, y1 + padding)
    if x1 <= x0 or y1 <= y0:
        return 0, 0, width, height
    return x0, y0, x1, y1


def bbox_from_mask(
    mask: np.ndarray, padding: int = 24
) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    h, w = mask.shape
    return clip_bbox(
        (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
        w,
        h,
        padding=padding,
    )


def copy_episode_metadata(src_episode: Path, dst_episode: Path, overwrite: bool = False):
    dst_episode.mkdir(parents=True, exist_ok=True)
    for p in src_episode.iterdir():
        if p.is_file() and p.suffix.lower() != ".jpg":
            out = dst_episode / p.name
            if overwrite or not out.exists():
                shutil.copy2(p, out)


@dataclass
class MaskStats:
    area: int
    area_ratio: float
    component_count: int
    threshold: float
    failed: bool
    reason: str = ""


def _otsu_threshold(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    hist, edges = np.histogram(values, bins=128)
    centers = (edges[:-1] + edges[1:]) * 0.5
    weight_left = np.cumsum(hist).astype(np.float64)
    weight_right = np.cumsum(hist[::-1]).astype(np.float64)[::-1]
    sum_left = np.cumsum(hist * centers).astype(np.float64)
    sum_right = np.cumsum((hist * centers)[::-1]).astype(np.float64)[::-1]
    mean_left = sum_left / np.maximum(weight_left, 1.0)
    mean_right = sum_right / np.maximum(weight_right, 1.0)
    score = weight_left[:-1] * weight_right[1:] * (
        mean_left[:-1] - mean_right[1:]
    ) ** 2
    if score.size == 0:
        return float(np.percentile(values, 60))
    return float(centers[:-1][int(np.argmax(score))])


def _component_texture(gray: np.ndarray) -> np.ndarray:
    gx = ndimage.sobel(gray, axis=1, mode="nearest")
    gy = ndimage.sobel(gray, axis=0, mode="nearest")
    return np.hypot(gx, gy)


def _select_glove_components(
    mask: np.ndarray,
    gray: np.ndarray,
    min_area: int,
    max_components: int,
    prev_mask_crop: Optional[np.ndarray],
) -> Tuple[np.ndarray, int]:
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask & False, 0
    areas = np.bincount(labels.ravel())
    texture = _component_texture(gray)
    h, w = mask.shape
    roi_center = np.array([w * 0.5, h * 0.42], dtype=np.float32)
    max_distance = math.hypot(w, h) + 1e-6
    components = []
    for label in range(1, count + 1):
        area = int(areas[label])
        if area < min_area:
            continue
        component = labels == label
        ys, xs = np.nonzero(component)
        if xs.size == 0:
            continue
        centroid = np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32)
        distance_score = 1.0 - min(
            float(np.linalg.norm(centroid - roi_center) / max_distance), 1.0
        )
        texture_score = float(np.mean(texture[component]))
        area_score = math.log1p(area)
        # Penalize components glued to ROI borders. Walls and light strips often
        # enter from the crop edge, while the annotated hand is usually inside
        # the crop or only touches the top edge.
        touches_left = xs.min() <= 1
        touches_right = xs.max() >= w - 2
        touches_bottom = ys.max() >= h - 2
        border_penalty = 0.0
        if touches_left:
            border_penalty += 1.0
        if touches_right:
            border_penalty += 1.0
        if touches_bottom:
            border_penalty += 1.5
        if prev_mask_crop is not None and prev_mask_crop.shape == mask.shape:
            overlap = int(((labels == label) & prev_mask_crop).sum())
        else:
            overlap = 0
        overlap_score = math.log1p(overlap) * 2.5
        score = (
            area_score * 1.25
            + distance_score * 8.0
            + min(texture_score / 18.0, 4.0)
            + overlap_score
            - border_penalty * 4.0
        )
        components.append((overlap > 0, score, area, label))
    if not components:
        return mask & False, count

    if prev_mask_crop is not None and any(c[0] for c in components):
        selected = sorted(
            [c for c in components if c[0]], key=lambda x: x[1], reverse=True
        )[:max_components]
    else:
        selected = sorted(components, key=lambda x: x[1], reverse=True)[:max_components]

    out = np.zeros_like(mask, dtype=bool)
    for _, _, _, label in selected:
        out |= labels == label
    return out, count


def segment_white_glove(
    image: np.ndarray,
    bbox: Sequence[int],
    exclude_bboxes: Optional[Sequence[Sequence[int]]] = None,
    prev_mask: Optional[np.ndarray] = None,
    min_area: int = 180,
    max_components: int = 1,
    bbox_padding: int = 0,
) -> Tuple[np.ndarray, MaskStats]:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = clip_bbox(bbox, w, h, padding=bbox_padding)
    crop = image[y0:y1, x0:x1].astype(np.float32)
    if crop.size == 0:
        return np.zeros((h, w), dtype=bool), MaskStats(0, 0.0, 0, 0.0, True, "empty_bbox")

    r, g, b = crop[..., 0], crop[..., 1], crop[..., 2]
    maxc = np.maximum.reduce([r, g, b])
    minc = np.minimum.reduce([r, g, b])
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    sat = (maxc - minc) / np.maximum(maxc, 1.0)

    # White fabric in this dataset is bright, low-saturation, and locally textured.
    score = luma + 0.35 * minc - 85.0 * sat
    otsu = _otsu_threshold(score.reshape(-1))
    percentile = float(np.percentile(score, 58))
    threshold = max(otsu, percentile)

    candidate = (
        (score >= threshold)
        & (luma >= 42.0)
        & (sat <= 0.58)
        & (minc >= 28.0)
    )

    for exclude_bbox in exclude_bboxes or []:
        ex0, ey0, ex1, ey1 = clip_bbox(exclude_bbox, w, h)
        rx0 = max(ex0, x0) - x0
        ry0 = max(ey0, y0) - y0
        rx1 = min(ex1, x1) - x0
        ry1 = min(ey1, y1) - y0
        if rx1 > rx0 and ry1 > ry0:
            candidate[ry0:ry1, rx0:rx1] = False

    candidate = ndimage.binary_opening(candidate, iterations=1)
    candidate = ndimage.binary_closing(candidate, iterations=2)
    candidate = ndimage.binary_fill_holes(candidate)

    prev_crop = None
    if prev_mask is not None and prev_mask.shape == (h, w):
        prev_crop = prev_mask[y0:y1, x0:x1]
        if prev_crop.any():
            prev_crop = ndimage.binary_dilation(prev_crop, iterations=8)

    selected, component_count = _select_glove_components(
        candidate,
        gray=luma,
        min_area=min_area,
        max_components=max_components,
        prev_mask_crop=prev_crop,
    )
    selected = ndimage.binary_closing(selected, iterations=2)
    selected = ndimage.binary_fill_holes(selected)

    full = np.zeros((h, w), dtype=bool)
    full[y0:y1, x0:x1] = selected
    area = int(full.sum())
    area_ratio = float(area / max(h * w, 1))
    failed = area < min_area
    reason = "too_small" if failed else ""
    return full, MaskStats(area, area_ratio, component_count, threshold, failed, reason)


def segment_human_hand(
    image: np.ndarray,
    bbox: Sequence[int],
    exclude_bboxes: Optional[Sequence[Sequence[int]]] = None,
    prev_mask: Optional[np.ndarray] = None,
    min_area: int = 180,
    max_components: int = 1,
    bbox_padding: int = 0,
) -> Tuple[np.ndarray, MaskStats]:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = clip_bbox(bbox, w, h, padding=bbox_padding)
    crop = image[y0:y1, x0:x1].astype(np.float32)
    if crop.size == 0:
        return np.zeros((h, w), dtype=bool), MaskStats(0, 0.0, 0, 0.0, True, "empty_bbox")

    r, g, b = crop[..., 0], crop[..., 1], crop[..., 2]
    maxc = np.maximum.reduce([r, g, b])
    minc = np.minimum.reduce([r, g, b])
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    sat = (maxc - minc) / np.maximum(maxc, 1.0)

    warm = (r > b * 0.92) & (g > b * 0.68) & (r > 25.0) & (g > 18.0)
    visible = (luma > 22.0) & (luma < 245.0) & (sat > 0.035)
    score = luma + 70.0 * sat + 45.0 * warm.astype(np.float32)
    threshold = max(_otsu_threshold(score.reshape(-1)), float(np.percentile(score, 45)))
    candidate = warm & visible & (score >= threshold)

    for exclude_bbox in exclude_bboxes or []:
        ex0, ey0, ex1, ey1 = clip_bbox(exclude_bbox, w, h)
        rx0 = max(ex0, x0) - x0
        ry0 = max(ey0, y0) - y0
        rx1 = min(ex1, x1) - x0
        ry1 = min(ey1, y1) - y0
        if rx1 > rx0 and ry1 > ry0:
            candidate[ry0:ry1, rx0:rx1] = False

    candidate = ndimage.binary_opening(candidate, iterations=1)
    candidate = ndimage.binary_closing(candidate, iterations=2)
    candidate = ndimage.binary_fill_holes(candidate)

    prev_crop = None
    if prev_mask is not None and prev_mask.shape == (h, w):
        prev_crop = prev_mask[y0:y1, x0:x1]
        if prev_crop.any():
            prev_crop = ndimage.binary_dilation(prev_crop, iterations=8)

    selected, component_count = _select_glove_components(
        candidate,
        gray=luma,
        min_area=min_area,
        max_components=max_components,
        prev_mask_crop=prev_crop,
    )
    selected = ndimage.binary_closing(selected, iterations=2)
    selected = ndimage.binary_fill_holes(selected)

    full = np.zeros((h, w), dtype=bool)
    full[y0:y1, x0:x1] = selected
    area = int(full.sum())
    area_ratio = float(area / max(h * w, 1))
    failed = area < min_area
    reason = "too_small" if failed else ""
    return full, MaskStats(area, area_ratio, component_count, threshold, failed, reason)


def feather_mask(mask: np.ndarray, radius: float = 4.0) -> np.ndarray:
    if not mask.any():
        return mask.astype(np.float32)
    inside = ndimage.distance_transform_edt(mask)
    outside = ndimage.distance_transform_edt(~mask)
    signed = inside - outside
    alpha = np.clip((signed + radius) / (2.0 * radius), 0.0, 1.0)
    return alpha.astype(np.float32)


@dataclass
class AugStyle:
    target_color: Tuple[float, float, float]
    mix: float
    brightness: float
    contrast: float
    texture_strength: float
    weave_freq_x: float
    weave_freq_y: float
    noise_strength: float
    rib_strength: float
    speckle_strength: float
    sheen_strength: float
    shade_contrast: float
    background_brightness: float
    background_contrast: float
    background_hue_shift: float
    background_saturation: float
    background_mix: float
    seed: int


PALETTES = [
    (64, 38, 28),
    (88, 55, 38),
    (102, 54, 42),
    (112, 72, 50),
    (132, 86, 58),
    (148, 76, 58),
    (155, 104, 72),
    (176, 124, 88),
    (194, 142, 102),
    (210, 158, 116),
    (224, 178, 138),
    (236, 194, 154),
    (244, 206, 168),
    (198, 132, 92),
    (174, 96, 104),
    (216, 132, 116),
]


def make_style(
    seed: int,
    variant: int = 0,
    glove_mix_range: Tuple[float, float] = (0.20, 0.55),
    glove_brightness_range: Tuple[float, float] = (0.90, 1.12),
    glove_contrast_range: Tuple[float, float] = (0.88, 1.18),
    glove_texture_strength_range: Tuple[float, float] = (0.0, 8.0),
    glove_noise_strength_range: Tuple[float, float] = (0.5, 4.0),
    glove_rib_strength_range: Tuple[float, float] = (0.0, 5.0),
    glove_speckle_strength_range: Tuple[float, float] = (0.0, 5.0),
    glove_sheen_strength_range: Tuple[float, float] = (0.0, 10.0),
    glove_shade_contrast_range: Tuple[float, float] = (0.85, 1.20),
    background_brightness_range: Tuple[float, float] = (0.92, 1.08),
    background_contrast_range: Tuple[float, float] = (0.92, 1.10),
    background_hue_shift_range: Tuple[float, float] = (-0.035, 0.035),
    background_saturation_range: Tuple[float, float] = (0.92, 1.08),
    background_mix_range: Tuple[float, float] = (1.0, 1.0),
) -> AugStyle:
    rng = random.Random(seed + 10007 * variant)
    color = tuple(float(v) for v in rng.choice(PALETTES))
    return AugStyle(
        target_color=color,
        mix=rng.uniform(*glove_mix_range),
        brightness=rng.uniform(*glove_brightness_range),
        contrast=rng.uniform(*glove_contrast_range),
        texture_strength=rng.uniform(*glove_texture_strength_range),
        weave_freq_x=rng.uniform(0.055, 0.095),
        weave_freq_y=rng.uniform(0.06, 0.11),
        noise_strength=rng.uniform(*glove_noise_strength_range),
        rib_strength=rng.uniform(*glove_rib_strength_range),
        speckle_strength=rng.uniform(*glove_speckle_strength_range),
        sheen_strength=rng.uniform(*glove_sheen_strength_range),
        shade_contrast=rng.uniform(*glove_shade_contrast_range),
        background_brightness=rng.uniform(*background_brightness_range),
        background_contrast=rng.uniform(*background_contrast_range),
        background_hue_shift=rng.uniform(*background_hue_shift_range),
        background_saturation=rng.uniform(*background_saturation_range),
        background_mix=rng.uniform(*background_mix_range),
        seed=seed + 10007 * variant,
    )


def _apply_background_color_transform(image: np.ndarray, style: AugStyle) -> np.ndarray:
    pil = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB")

    if (
        abs(style.background_hue_shift) > 1e-6
        or abs(style.background_saturation - 1.0) > 1e-3
    ):
        hsv = np.asarray(pil.convert("HSV")).astype(np.int16)
        if abs(style.background_hue_shift) > 1e-6:
            hue_delta = int(round(style.background_hue_shift * 255.0))
            hsv[..., 0] = (hsv[..., 0] + hue_delta) % 256
        if abs(style.background_saturation - 1.0) > 1e-3:
            hsv[..., 1] = np.clip(
                hsv[..., 1].astype(np.float32) * style.background_saturation,
                0,
                255,
            ).astype(np.int16)
        pil = Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")

    if abs(style.background_brightness - 1.0) > 1e-3:
        pil = ImageEnhance.Brightness(pil).enhance(style.background_brightness)
    if abs(style.background_contrast - 1.0) > 1e-3:
        pil = ImageEnhance.Contrast(pil).enhance(style.background_contrast)

    return np.asarray(pil).astype(np.float32)


def apply_glove_augmentation(
    image: np.ndarray,
    mask: np.ndarray,
    style: AugStyle,
    frame_index: int,
    feather_radius: float = 4.0,
) -> np.ndarray:
    base = image.astype(np.float32)
    h, w = mask.shape
    inside_alpha_2d = (
        feather_mask(mask, radius=feather_radius)
        if mask.any()
        else np.zeros((h, w), dtype=np.float32)
    )
    outside_alpha = np.clip(
        (1.0 - inside_alpha_2d)[..., None] * style.background_mix,
        0.0,
        1.0,
    )
    background_adjusted = _apply_background_color_transform(base, style)
    background_out = base * (1.0 - outside_alpha) + background_adjusted * outside_alpha

    if not mask.any():
        return np.clip(background_out, 0, 255).astype(np.uint8)

    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    luma = (
        0.299 * base[..., 0] + 0.587 * base[..., 1] + 0.114 * base[..., 2]
    ) / 255.0

    rng = np.random.default_rng(style.seed + frame_index * 7919)
    noise = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
    noise = ndimage.gaussian_filter(noise, sigma=1.0)
    fine_lines = (
        np.sin(xx * style.weave_freq_x * 1.7 + style.seed * 0.013)
        + 0.6 * np.sin(yy * style.weave_freq_y * 1.3 + style.seed * 0.019)
        + 0.35 * np.sin((xx - yy) * style.weave_freq_x * 2.4 + style.seed * 0.007)
    )
    blotch = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
    blotch = ndimage.gaussian_filter(blotch, sigma=18.0)
    blotch = blotch / max(float(np.std(blotch)), 1e-6)
    speckle_seed = rng.random((h, w), dtype=np.float32)
    speckle_density = np.clip(0.992 - style.speckle_strength * 0.006, 0.90, 0.992)
    dark_speckle = (speckle_seed > speckle_density).astype(np.float32)
    light_speckle = (speckle_seed < (1.0 - speckle_density) * 0.45).astype(np.float32)
    speckle = ndimage.gaussian_filter(dark_speckle - 0.55 * light_speckle, sigma=0.45)
    speckle = speckle * 2.2 - 0.02
    sheen = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
    sheen = ndimage.gaussian_filter(sheen, sigma=7.0)
    sheen = np.maximum(sheen - float(np.percentile(sheen, 62.0)), 0.0)
    sheen = sheen / max(float(sheen.max()), 1e-6)

    texture = (
        style.texture_strength * fine_lines * 1.15
        + style.noise_strength * noise * 1.25
        + style.texture_strength * blotch * 0.65
        + style.rib_strength * np.sin((xx + yy) * 0.045 + style.seed * 0.005)
        + style.speckle_strength * speckle * 1.35
        + style.sheen_strength * sheen * 1.45
    )
    color_blotch = rng.normal(0.0, 1.0, size=(h, w, 3)).astype(np.float32)
    for channel in range(3):
        color_blotch[..., channel] = ndimage.gaussian_filter(
            color_blotch[..., channel],
            sigma=22.0,
        )
    color_blotch = color_blotch / max(float(np.std(color_blotch)), 1e-6)
    color = np.asarray(style.target_color, dtype=np.float32)
    current = base[mask]
    current_mean = current.mean(axis=0) if current.size else color
    tint_strength = np.clip(style.mix * 0.85, 0.0, 0.85)
    hand = base + (color - current_mean)[None, None, :] * tint_strength
    channel_scale = np.asarray([1.12, 0.82, 0.72], dtype=np.float32)
    hand = hand + color_blotch * (style.texture_strength * 0.55) * channel_scale[None, None, :]

    mask_luma = luma[mask]
    center_luma = float(mask_luma.mean()) if mask_luma.size else 0.5
    local_shade = 1.0 + (luma - center_luma) * (style.shade_contrast - 1.0) * 1.25
    hand = (hand - 127.5) * style.contrast + 127.5
    hand = hand * style.brightness * np.clip(local_shade[..., None], 0.55, 1.55)
    hand = hand + texture[..., None]

    alpha = np.clip(inside_alpha_2d[..., None] * style.mix * 1.15, 0.0, 1.0)
    out = background_out * (1.0 - alpha) + np.clip(hand, 0, 255) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def make_overlay(
    original: np.ndarray,
    augmented: np.ndarray,
    mask: np.ndarray,
    bbox: Optional[Sequence[int]] = None,
    exclude_bboxes: Optional[Sequence[Sequence[int]]] = None,
) -> Image.Image:
    orig = Image.fromarray(original.astype(np.uint8), mode="RGB")
    aug = Image.fromarray(augmented.astype(np.uint8), mode="RGB")
    overlay = orig.copy()
    red = Image.new("RGB", overlay.size, (255, 40, 80))
    alpha = Image.fromarray((feather_mask(mask, radius=2.0) * 130).astype(np.uint8), mode="L")
    overlay = Image.composite(red, overlay, alpha)

    if bbox is not None:
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(tuple(normalize_bbox(bbox)), outline=(0, 255, 255), width=2)
        for exclude_bbox in exclude_bboxes or []:
            draw.rectangle(tuple(normalize_bbox(exclude_bbox)), outline=(255, 180, 0), width=2)

    gap = 8
    canvas = Image.new(
        "RGB", (orig.width * 3 + gap * 2, orig.height), (20, 20, 20)
    )
    canvas.paste(orig, (0, 0))
    canvas.paste(overlay, (orig.width + gap, 0))
    canvas.paste(aug, ((orig.width + gap) * 2, 0))
    return canvas


def write_json(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
    tmp.replace(path)


def evenly_spaced_indices(length: int, count: int) -> List[int]:
    if length <= 0 or count <= 0:
        return []
    if length <= count:
        return list(range(length))
    return sorted(set(int(round(v)) for v in np.linspace(0, length - 1, count)))


def episode_seed(base_seed: int, episode_name: str) -> int:
    total = base_seed
    for ch in episode_name:
        total = (total * 131 + ord(ch)) % (2**31 - 1)
    return total


def annotation_bbox_for_episode(annotation_data: Dict, episode_name: str):
    item = annotation_data.get("episodes", {}).get(episode_name)
    if not item:
        return None
    return normalize_bbox(item["bbox"])


def annotation_for_episode(annotation_data: Dict, episode_name: str):
    item = annotation_data.get("episodes", {}).get(episode_name)
    if not item:
        return None, []
    bbox = normalize_bbox(item["bbox"])
    exclude_bboxes = [normalize_bbox(b) for b in item.get("exclude_bboxes", [])]
    return bbox, exclude_bboxes


def summarize_style(style: AugStyle) -> Dict:
    return {
        "target_color": [round(v, 3) for v in style.target_color],
        "mix": round(style.mix, 4),
        "brightness": round(style.brightness, 4),
        "contrast": round(style.contrast, 4),
        "texture_strength": round(style.texture_strength, 4),
        "weave_freq_x": round(style.weave_freq_x, 5),
        "weave_freq_y": round(style.weave_freq_y, 5),
        "noise_strength": round(style.noise_strength, 4),
        "rib_strength": round(style.rib_strength, 4),
        "speckle_strength": round(style.speckle_strength, 4),
        "sheen_strength": round(style.sheen_strength, 4),
        "shade_contrast": round(style.shade_contrast, 4),
        "background_brightness": round(style.background_brightness, 4),
        "background_contrast": round(style.background_contrast, 4),
        "background_hue_shift": round(style.background_hue_shift, 5),
        "background_saturation": round(style.background_saturation, 4),
        "background_mix": round(style.background_mix, 4),
        "seed": style.seed,
    }
