import numpy as np


class U:
    @staticmethod
    def unit(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v) + 1e-8
        return v / n

    @staticmethod
    def limit_speed(v: np.ndarray, vmax: float) -> np.ndarray:
        s = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
        f = np.minimum(1.0, vmax / s)
        return v * f

    @staticmethod
    def speed_profile(dist: float, v_min: float, v_max: float, slow_radius: float) -> float:
        if dist >= slow_radius:
            return v_max
        ratio = dist / max(slow_radius, 1e-6)
        return v_min + (v_max - v_min) * ratio

    @staticmethod
    def accel_to_target_velocity(v_current: np.ndarray, v_target: np.ndarray, k: float) -> np.ndarray:
        return k * (v_target - v_current)
