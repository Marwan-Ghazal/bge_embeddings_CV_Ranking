# calibration.py
import os, pickle, numpy as np

def _piecewise_default(x: float) -> float:
    """
    Monotonic piecewise linear mapping that widens 0.30–0.60 into ~0.40–0.93.
    Tweak points to taste; ordering is preserved.
    """
    # (raw -> calibrated) in [0,1]
    pts = [
        (0.00, 0.05),
        (0.30, 0.40),  # bad ~30% -> 40%
        (0.40, 0.58),
        (0.45, 0.68),  # borderline ~45% -> 68%
        (0.50, 0.80),  # good ~50% -> 80%
        (0.55, 0.88),
        (0.60, 0.93),
        (0.70, 0.96),
        (0.80, 0.985),
        (1.00, 1.00),
    ]
    x = float(np.clip(x, 0.0, 1.0))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            t = (x - x0) / max(1e-6, (x1 - x0))
            return float(y0 + t * (y1 - y0))
    return 1.0

def calibrate(raw: float, iso_path: str = "calibration_isotonic.pkl"):
    """
    Try isotonic (if you’ve trained one offline); else use the piecewise default.
    Returns (calibrated_0_1, kind).
    """
    # Optional learned calibrator
    try:
        if os.path.exists(iso_path):
            with open(iso_path, "rb") as f:
                iso = pickle.load(f)  # e.g., sklearn IsotonicRegression
            y = iso.predict([float(raw)])[0]
            return float(np.clip(y, 0.0, 1.0)), "isotonic"
    except Exception:
        pass
    # Fallback
    return _piecewise_default(float(raw)), "piecewise-default"
