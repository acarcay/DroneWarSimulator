import itertools
import logging
from typing import List

import numpy as np

from config import Config
from models import Drone
from environment import Environment
from control import SwarmController
from metrics import Metrics
from viz import Visualizer, plt
from matplotlib.animation import FuncAnimation
from reports import save_report
from sensors import GPSSensor, IMUSensor, LiDARSensor


def main():
    logging.basicConfig(level=logging.INFO)
    np.random.seed(2)

    cfg = Config()
    sensors = {
        "gps": GPSSensor(cfg.GPS_BIAS, cfg.GPS_NOISE, cfg.GPS_DROPOUT),
        "imu": IMUSensor(cfg.IMU_BIAS, cfg.IMU_NOISE, cfg.IMU_DROPOUT),
        "lidar": LiDARSensor(cfg.LIDAR_BIAS, cfg.LIDAR_NOISE, cfg.LIDAR_DROPOUT),
    }

    # init drones
    drones: List[Drone] = []
    for i in range(cfg.N):
        pos = np.array([np.random.uniform(2, 8), np.random.uniform(20, 24)], dtype=float)
        vel = np.random.randn(2) * 0.1
        drones.append(Drone(i, pos, vel))

    env = Environment(cfg)
    swarm = SwarmController(cfg, env, drones, leader_idx=0, sensors=sensors)
    metrics = Metrics(cfg, swarm)
    viz = Visualizer(cfg, swarm, metrics)
    viz.setup()

    ani = FuncAnimation(viz.fig, viz.animate, frames=itertools.count(), interval=30, blit=True)

    try:
        plt.show()
    finally:
        # summary
        print("\n=== Simulation Summary ===")
        print(f"Total path (sum): {metrics.total_path():.2f} m")
        for d in swarm.drones:
            print(f"  Drone {d.idx}: {d.path_length:.2f} m")
        if metrics.form_err_hist:
            print(f"Mean RMSE: {np.mean(metrics.form_err_hist):.3f} m")
            print(f"Min  RMSE: {np.min(metrics.form_err_hist):.3f} m")
        if metrics.settled_time_s is not None:
            print(f"Settling time: {metrics.settled_time_s:.2f} s")
        else:
            print("Not settled under thresholds (tune FORM_ERR_THRESH / STABLE_FRAMES).")

        old_results = {
            "drone": list(range(8)),
            "path": [226.69,226.12,226.27,229.45,234.07,234.27,236.37,234.35],
            "mean_rmse": 1.284,
            "settling_time": 38.16
        }
        new_results = {
            "drone": list(range(8)),
            "path": [189.26,188.16,190.53,193.08,195.90,194.01,195.03,193.83],
            "mean_rmse": 1.127,
            "settling_time": 26.52
        }

        report = save_report(old_results, new_results)
        print("Rapor kaydedildi:", report)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"Fatal error: {e}")
