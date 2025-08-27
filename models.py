from dataclasses import dataclass
import numpy as np


@dataclass
class Drone:
    idx: int
    pos: np.ndarray
    vel: np.ndarray
    path_length: float = 0.0

    def update_position(self, dt: float) -> None:
        self.pos = self.pos + self.vel * dt

    def add_path(self, prev_pos: np.ndarray) -> None:
        self.path_length += float(np.linalg.norm(self.pos - prev_pos))
