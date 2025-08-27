import logging
from typing import List, Optional

import numpy as np

from config import Config
from control import SwarmController


class Metrics:
    def __init__(self, cfg: Config, swarm: SwarmController):
        self.cfg = cfg
        self.swarm = swarm
        self.form_err_hist: List[float] = []
        self.stable_count: int = 0
        self.settled_time_s: Optional[float] = None
        self.frames: int = 0

    def formation_rmse(self) -> float:
        pos = self.swarm.positions()
        vel = self.swarm.velocities()
        desired = np.zeros_like(pos)
        leader = self.swarm.leader
        # Takipçiler anchor + slot'u hedefler; lider için hedef koymuyoruz (0 katkı)
        anchor = self.swarm.leader_anchor(pos, vel)
        desired[leader] = pos[leader]
        for i in range(self.cfg.N):
            if i == leader:
                continue
            slot = self.swarm.assignments[i]
            desired[i] = anchor + self.swarm.OFFSETS[slot]
        diffs = pos[1:] - desired[1:]
        rmse = float(np.sqrt(np.mean(np.sum(diffs**2, axis=1))))
        return rmse

    def min_separation(self) -> float:
        pos = self.swarm.positions()
        dmat = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        with np.errstate(invalid='ignore'):
            dmat[dmat == 0] = np.nan
        return float(np.nanmin(dmat))

    def update(self) -> None:
        rmse = self.formation_rmse()
        self.form_err_hist.append(rmse)
        min_sep = self.min_separation()
        if (rmse < self.cfg.FORM_ERR_THRESH) and (min_sep > self.cfg.D_SAFE):
            self.stable_count += 1
            if self.settled_time_s is None and self.stable_count >= self.cfg.STABLE_FRAMES:
                self.settled_time_s = self.frames * self.cfg.DT
                logging.info(f"[SETTLED] t={self.settled_time_s:.2f}s, rmse={rmse:.2f}, min_sep={min_sep:.2f}")
        else:
            self.stable_count = 0
        self.frames += 1

    def total_path(self) -> float:
        return float(sum(d.path_length for d in self.swarm.drones))
