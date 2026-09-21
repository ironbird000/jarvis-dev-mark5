from typing import Any, Dict, Optional

_CANONICAL_TARGET_MODES = {"point", "region"}
_TARGET_MODE_ALIASES = {
    "crosshair": "point",
    "point": "point",
    "box": "region",
    "region": "region",
}


def canonicalize_target_mode(value: Any) -> Optional[str]:
    mode = str(value or "").strip().lower()
    if not mode:
        return None
    return _TARGET_MODE_ALIASES.get(mode)


def infer_target_mode(
    *,
    target_mode: Any,
    target_point: Optional[Dict[str, Any]],
    target_region: Optional[Dict[str, Any]],
    default_mode: str = "point",
) -> str:
    canonical = canonicalize_target_mode(target_mode)
    if canonical:
        return canonical
    if isinstance(target_point, dict):
        return "point"
    if isinstance(target_region, dict):
        region_mode = canonicalize_target_mode(target_region.get("mode"))
        if region_mode:
            return region_mode
        return "region"
    return default_mode if default_mode in _CANONICAL_TARGET_MODES else "point"


def _coordinate_space(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return "normalized"
    if raw in {"normalized", "norm"}:
        return "normalized"
    if raw in {"pixels", "pixel", "px"}:
        return "pixels"
    return ""


def _validate_normalized_value(value: Any) -> bool:
    try:
        coordinate = float(value)
    except (TypeError, ValueError):
        return False
    return 0.0 <= coordinate <= 1.0


def validate_target_payload(
    *,
    target_mode: str,
    target_point: Optional[Dict[str, Any]],
    target_region: Optional[Dict[str, Any]],
) -> None:
    if target_mode == "point":
        if not isinstance(target_point, dict):
            return
        space = _coordinate_space(target_point.get("coordinate_space") or target_point.get("space") or target_point.get("units"))
        if not space:
            raise ValueError("target_point coordinate_space must be normalized or pixels")
        if space == "normalized":
            if not _validate_normalized_value(target_point.get("x")) or not _validate_normalized_value(target_point.get("y")):
                raise ValueError("target_point normalized coordinates must be between 0.0 and 1.0")
        return

    if target_mode == "region":
        if not isinstance(target_region, dict):
            return
        space = _coordinate_space(target_region.get("coordinate_space") or target_region.get("space") or target_region.get("units"))
        if not space:
            raise ValueError("target_region coordinate_space must be normalized or pixels")
        try:
            width = float(target_region.get("width"))
            height = float(target_region.get("height"))
        except (TypeError, ValueError):
            raise ValueError("target_region width and height must be numeric")
        if width <= 0.0 or height <= 0.0:
            raise ValueError("target_region width and height must be greater than 0")
        if space == "normalized":
            if not _validate_normalized_value(target_region.get("x")):
                raise ValueError("target_region normalized x must be between 0.0 and 1.0")
            if not _validate_normalized_value(target_region.get("y")):
                raise ValueError("target_region normalized y must be between 0.0 and 1.0")
            if width > 1.0 or height > 1.0:
                raise ValueError("target_region normalized width/height must be between 0.0 and 1.0")


def normalize_coordinate(value: Any, *, max_dimension: int, coordinate_space: str) -> Optional[float]:
    try:
        coordinate = float(value)
    except (TypeError, ValueError):
        return None
    if coordinate_space == "normalized":
        if coordinate < 0.0 or coordinate > 1.0:
            return None
        return coordinate
    if coordinate_space == "pixels":
        if max_dimension <= 0:
            return None
        return coordinate / float(max_dimension)
    return None


def coordinate_space_for_payload(target: Dict[str, Any]) -> str:
    return _coordinate_space(target.get("coordinate_space") or target.get("space") or target.get("units"))


def target_region_from_point(target_point: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(target_point, dict):
        return {}
    coordinate_space = target_point.get("coordinate_space") or target_point.get("space") or target_point.get("units")
    space = _coordinate_space(coordinate_space) or "normalized"
    footprint = 2.0 if space == "pixels" else 0.01
    center_x = target_point.get("x")
    center_y = target_point.get("y")
    try:
        region_x = float(center_x) - (footprint / 2.0)
    except (TypeError, ValueError):
        region_x = center_x
    try:
        region_y = float(center_y) - (footprint / 2.0)
    except (TypeError, ValueError):
        region_y = center_y
    region = {
        "x": region_x,
        "y": region_y,
        "width": footprint,
        "height": footprint,
        "mode": "crosshair",
        "center_x": center_x,
        "center_y": center_y,
    }
    if coordinate_space is not None:
        region["coordinate_space"] = coordinate_space
    return region
