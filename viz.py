from typing import Tuple

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from config import Config
from control import SwarmController
from metrics import Metrics


class Visualizer:
    def __init__(self, cfg: Config, swarm: SwarmController, metrics: Metrics):
        self.cfg = cfg
        self.swarm = swarm
        self.metrics = metrics
        self.paused = False
        self.fig, self.ax = plt.subplots(figsize=cfg.FIGSIZE)
        self.scat = None
        self.goal_line, = self.ax.plot([], [], 'r:', lw=1, zorder=2)

    def setup(self) -> None:
        self.ax.set_xlim(self.cfg.WORLD[0], self.cfg.WORLD[1])
        self.ax.set_ylim(self.cfg.WORLD[2], self.cfg.WORLD[3])
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.6)

        # obstacles
        for (cx, cy, r) in self.cfg.OBST:
            self.ax.add_patch(Circle((cx, cy), r, color='gray', alpha=0.7, zorder=1))
            self.ax.add_patch(Circle((cx, cy), r+self.cfg.OBS_INFLUENCE, color='gray', alpha=0.1, zorder=0))

        # waypoints
        W = self.cfg.WAYPOINTS
        self.ax.plot(W[:,0], W[:,1], 'go--', label='Waypoints', zorder=2)
        self.ax.plot(W[0,0], W[0,1], 'g*', markersize=12, zorder=3)
        self.ax.legend(loc='upper right')

        # drones
        pos = self.swarm.positions()
        cols = ['red'] + ['blue']*(self.cfg.N-1)
        self.scat = self.ax.scatter(pos[:,0], pos[:,1], c=cols, zorder=3)
        self.title = self.ax.set_title('SwarmSim')

        # input
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def on_key(self, event) -> None:
        if event.key == ' ':
            self.paused = not self.paused
        elif event.key == 'g':
            self.swarm.mode = 'obstacle' if self.swarm.mode == 'waypoint' else 'waypoint'
        elif event.key == 'tab':
            self.swarm.sel_obs = (self.swarm.sel_obs + 1) % len(self.cfg.OBST)
        elif event.key == 'backspace':
            self.swarm.sel_obs = (self.swarm.sel_obs - 1) % len(self.cfg.OBST)

    def hud(self) -> None:
        total_path = self.metrics.total_path()
        current_rmse = self.metrics.form_err_hist[-1] if self.metrics.form_err_hist else 0.0
        st = f"{self.metrics.settled_time_s:.2f}s" if self.metrics.settled_time_s is not None else "—"
        self.title.set_text(
            f"Mode: {self.swarm.mode.capitalize()} | TotalPath: {total_path:.1f} m | RMSE: {current_rmse:.2f} m | Settle: {st}"
        )

    def redraw_goal_line(self) -> None:
        g = self.swarm.current_goal()
        pos = self.swarm.positions()
        L = self.swarm.leader
        self.goal_line.set_data([pos[L,0], g[0]], [pos[L,1], g[1]])

    def draw(self) -> Tuple:
        pos = self.swarm.positions()
        self.scat.set_offsets(pos)
        self.redraw_goal_line()
        self.hud()
        return self.scat, self.goal_line

    def animate(self, _):
        if not self.paused:
            self.swarm.step()
            self.metrics.update()
        return self.draw()
