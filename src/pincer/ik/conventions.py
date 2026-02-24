"""Motor <-> URDF angle convention mappings for the XLerobot arm and head."""

import numpy as np


def arm_motor_to_urdf(q_deg: np.ndarray) -> np.ndarray:
    """Convert arm joint angles from motor convention to URDF convention (degrees)."""
    out = q_deg.copy()
    out[1] = 90.0 - out[1]
    out[2] = out[2] + 90.0
    return out


def arm_urdf_to_motor(q_deg: np.ndarray) -> np.ndarray:
    """Convert arm joint angles from URDF convention to motor convention (degrees)."""
    out = q_deg.copy()
    out[1] = 90.0 - out[1]
    out[2] = out[2] - 90.0
    return out


def head_motor_to_urdf(q_deg: np.ndarray) -> np.ndarray:
    """Convert head joint angles from motor convention to URDF convention (degrees).

    Pan axis is negated for correct left/right camera mapping.
    Tilt has an offset: motor zero is not URDF zero (horizontal).
    """
    out = q_deg.copy()
    out[0] = -out[0]
    out[1] = out[1] - 11.0
    return out
