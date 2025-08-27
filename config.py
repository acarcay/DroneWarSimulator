from dataclasses import dataclass, field
from typing import Tuple
import numpy as np


@dataclass
class Config:
    # world
    WORLD: np.ndarray = field(default_factory=lambda: np.array([0.0, 40.0, 0.0, 25.0], dtype=float))
    OBST: np.ndarray = field(default_factory=lambda: np.array([
        [12.0, 12.0, 2.5],
        [24.0,  8.0, 2.0],
        [30.0, 17.0, 3.0]
    ], dtype=float))
    OBS_INFLUENCE: float = 5.0

    # --- Early-slow / TTC tabanlı fren ---
    SAFE_PAD: float = 0.6       # engel/duvar güvenlik payı (m)
    TTC_SLOW: float = 2.5       # bu kadar saniyelik TTC hedefi (yumuşak yavaşlama)
    TTC_STOP: float = 1.2       # kritik TTC (sert fren başlar)
    K_BRAKE: float = 2.0        # fren ivme kazancı
    W_BRAKE: float = 1.0        # fren terimi ağırlığı

    R_FORM_MAX = 2.4            # açık alanda hedeflenen yarıçap
    R_FORM_ADAPT_ALPHA = 0.12   # yumuşatma katsayısı (0..1)
    ASSIGN_FREEZE_GAP = 2.0     # çok dar alanda slot atamasını dondur

    # slot hedefi LPF
    SLOT_LPF_ALPHA: float = 0.25

    COHESION_BETA: float = 0.10   # 0.08–0.14 aralığı iyi

    # Yanal kaçınma + bypass
    W_TANG: float = 1.2      # tangansiyel alan ağırlığı
    TANG_GAIN: float = 1.0   # engel çevresinde akış gücü
    BYPASS_CLEAR: float = 5.5  # bu mesafeden kısa TTC-ray varsa bypass başlat
    BYPASS_MARGIN: float = 0.8 # engel yarıçapına eklenecek güvenlik payı

    K_FORM_P = 2.2   # konum kazancı
    K_FORM_D = 1.1   # hız kazancı

    # swarm
    N: int = 8
    DT: float = 0.06
    MAX_SPEED: float = 4.0
    ACC_MAX: float = 12.0
    DAMPING: float = 0.06
    R_NEIGH: float = 4.0
    MARGIN: float = 2.0
    D_MIN: float = 1.2

    # behavior weights
    W_SEP: float = 0.65
    W_ALI: float = 0.7
    W_COH: float = 0.3
    W_GOAL: float = 2.0
    W_OBS: float = 1.0
    W_FORM: float = 1.2
    TURN_RATE_MAX: float = 0.25

    # adaptive speed
    V_MAX_FAR: float = 4.0
    V_MIN_NEAR: float = 0.6
    SLOW_RADIUS: float = 8.0
    K_VEL_TRACK: float = 1.9

    # waypoints
    WAYPOINTS: np.ndarray = field(default_factory=lambda: np.array([[6,4],[18,6],[34,20],[8,20],[36,8]], dtype=float))

    # assignment
    ASSIGN_PERIOD: int = 12    # frames between Hungarian re-assignments
    SWITCH_PENALTY: float = 2.0  # hysteresis to reduce thrash (0..3 advisable)

    # metrics
    FORM_ERR_THRESH: float = 1.2
    STABLE_FRAMES: int = 120
    D_SAFE: float = 1.0

    # vis
    FIGSIZE: Tuple[int, int] = (12, 7)

    # lead-ahead (adaptive)
    ADAPTIVE_LEAD: bool = True
    LEAD_MIN: float = 0.15
    LEAD_MAX: float = 0.60
    LEAD_PROX: float = 5.0   # engellere 0..5m yakınlıkta lead’i kıs
    LEAD_AHEAD_TAU: float = 0.4  # ADAPTIVE_LEAD=False olursa kullanılır

    # additional config
    R_SEP: float = 1.6        # sadece bu menzilde separation uygula
    OBS_FORCE_MAX: float = 12.0  # 6.0 -> 12.0: engelde “yetişsin”

    # sensors
    GPS_BIAS: np.ndarray = field(default_factory=lambda: np.zeros(2))  # GPS bias (m)
    GPS_NOISE: float = 0.0                                           # GPS noise std dev (m)
    GPS_DROPOUT: float = 0.0                                         # GPS dropout probability
    IMU_BIAS: np.ndarray = field(default_factory=lambda: np.zeros(2))  # IMU bias (m/s)
    IMU_NOISE: float = 0.0                                            # IMU noise std dev (m/s)
    IMU_DROPOUT: float = 0.0                                          # IMU dropout probability
    LIDAR_BIAS: np.ndarray = field(default_factory=lambda: np.zeros(2))  # LiDAR bias for obstacle vectors
    LIDAR_NOISE: float = 0.0                                           # LiDAR noise std dev
    LIDAR_DROPOUT: float = 0.0                                         # LiDAR dropout probability
