# ──────────────────────────────────────────────────────────────────────────────
# SwarmSim (Modular Refactor)
# Folder-style single file for convenience. You can split into modules as noted.
# ──────────────────────────────────────────────────────────────────────────────
# Files (suggested):
#   config.py, utils.py, models.py, environment.py, control.py, metrics.py, viz.py, main.py
# This single file contains all of them inline. Copy each section into its file.
# ──────────────────────────────────────────────────────────────────────────────

import json
import math
import itertools
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import heapq
from pymavlink import mavutil
import heapq, random, time
from comms.channel import LossyChannel
from fastapi import FastAPI, WebSocket
from reports import save_report

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

try:
    from scipy.optimize import linear_sum_assignment
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False

# ──────────────────────────────────────────────────────────────────────────────
# config.py
# ──────────────────────────────────────────────────────────────────────────────
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
    FIGSIZE: Tuple[int,int] = (12, 7)

    # lead-ahead (adaptive)
    ADAPTIVE_LEAD: bool = True
    LEAD_MIN: float = 0.15
    LEAD_MAX: float = 0.60
    LEAD_PROX: float = 5.0   # engellere 0..5m yakınlıkta lead’i kıs
    LEAD_AHEAD_TAU: float = 0.4  # ADAPTIVE_LEAD=False olursa kullanılır

    # additional config
    R_SEP: float = 1.6        # sadece bu menzilde separation uygula
    OBS_FORCE_MAX: float = 12.0  # 6.0 -> 12.0: engelde “yetişsin”

# ──────────────────────────────────────────────────────────────────────────────
# utils.py
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# models.py
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# environment.py
# ──────────────────────────────────────────────────────────────────────────────
class Environment:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def wall_repulsion(self, p: np.ndarray) -> np.ndarray:
        f = np.zeros(2)
        x0, x1, y0, y1 = self.cfg.WORLD
        m = self.cfg.MARGIN
        if p[0] < x0 + m: f[0] += (x0 + m - p[0])
        if p[0] > x1 - m: f[0] -= (p[0] - (x1 - m))
        if p[1] < y0 + m: f[1] += (y0 + m - p[1])
        if p[1] > y1 - m: f[1] -= (p[1] - (y1 - m))
        return f

    def obstacle_repulsion(self, p: np.ndarray) -> np.ndarray:
        """
        Uzakta yumuşak, engel kabuğuna yaklaşınca sertleşen; spike'ı sınırlı bir itiş.
        """
        f = np.zeros(2)
        for cx, cy, r in self.cfg.OBST:
            c = np.array([cx, cy], dtype=float)
            vec = p - c
            dist = np.linalg.norm(vec) + 1e-8
            R = r + self.cfg.OBS_INFLUENCE
            if dist < R:
                buffer = (r + 0.2) - dist   # içeri sızma pozitif
                gain = 1.0 + max(0.0, buffer) * 10.0
                f += gain * (vec / ((dist - r + 0.3) ** 2))
        n = np.linalg.norm(f)
        max_force = self.cfg.OBS_FORCE_MAX
        if n > max_force:
            f = (f / (n + 1e-8)) * max_force
        return f

    # ---- No-penetration collision resolution (post-step) ----
    def resolve_circle_collisions(self, p: np.ndarray, v: np.ndarray, pad: float = 0.05):
        p_corr, v_corr = p.copy(), v.copy()
        for cx, cy, r in self.cfg.OBST:
            c = np.array([cx, cy], dtype=float)
            vec = p_corr - c
            d = np.linalg.norm(vec) + 1e-8
            r_eff = r + pad
            if d < r_eff:
                n = vec / d
                p_corr = c + n * r_eff                 # pozisyonu kabuk üzerine it
                v_corr = v_corr - np.dot(v_corr, n) * n  # normal hız bileşenini sıfırla (sliding)
        return p_corr, v_corr

    def resolve_wall_collisions(self, p: np.ndarray, v: np.ndarray, pad: float = 0.05):
        x0, x1, y0, y1 = self.cfg.WORLD
        px, py = p.copy()
        vx, vy = v.copy()
        if px < x0 + pad:
            px = x0 + pad; vx = 0.0
        if px > x1 - pad:
            px = x1 - pad; vx = 0.0
        if py < y0 + pad:
            py = y0 + pad; vy = 0.0
        if py > y1 - pad:
            py = y1 - pad; vy = 0.0
        return np.array([px, py], dtype=float), np.array([vx, vy], dtype=float)

    def resolve_collisions(self, p: np.ndarray, v: np.ndarray):
        p1, v1 = self.resolve_circle_collisions(p, v, pad=0.05)
        p2, v2 = self.resolve_wall_collisions(p1, v1, pad=0.05)
        return p2, v2

    @staticmethod
    def _ray_circle_intersect(p: np.ndarray, d: np.ndarray, c: np.ndarray, r: float):
        # |(p - c) + t d| = r
        m = p - c
        b = np.dot(m, d)
        c0 = np.dot(m, m) - r*r
        disc = b*b - c0
        if disc < 0.0:
            return None
        sqrt_disc = math.sqrt(disc)
        t1 = -b - sqrt_disc
        t2 = -b + sqrt_disc
        ts = [t for t in (t1, t2) if t > 0.0]
        return min(ts) if ts else None

    def forward_clearance(self, p: np.ndarray, d: np.ndarray, pad: float) -> float:
        """
        p'den d yönünde (birim vektör) en yakın duvar/engel kabuğuna olan mesafe.
        """
        d = d / (np.linalg.norm(d) + 1e-8)
        hits = []

        # Duvarlar (x=x0+pad, x=x1-pad, y=y0+pad, y=y1-pad)
        x0, x1, y0, y1 = self.cfg.WORLD
        # x düzlemleri
        if d[0] > 1e-8:
            t = (x1 - pad - p[0]) / d[0]
            if t > 0: hits.append(t)
        if d[0] < -1e-8:
            t = (x0 + pad - p[0]) / d[0]
            if t > 0: hits.append(t)
        # y düzlemleri
        if d[1] > 1e-8:
            t = (y1 - pad - p[1]) / d[1]
            if t > 0: hits.append(t)
        if d[1] < -1e-8:
            t = (y0 + pad - p[1]) / d[1]
            if t > 0: hits.append(t)

        # Daire engeller
        for cx, cy, r in self.cfg.OBST:
            t = self._ray_circle_intersect(p, d, np.array([cx, cy], dtype=float), r + pad)
            if t is not None:
                hits.append(t)

        if not hits:
            return 1e6  # çok uzak: pratikte sonsuz
        return float(min(hits))
    
    def nearest_obstacle_on_ray(self, p: np.ndarray, d: np.ndarray, pad: float):
        """Ray üzerinde ilk kesişen engeli (merkez, r_eff, t) döndür; yoksa None."""
        d = d / (np.linalg.norm(d) + 1e-8)
        best = None; tbest = 1e9
        for cx, cy, r in self.cfg.OBST:
            t = self._ray_circle_intersect(p, d, np.array([cx, cy], float), r + pad)
            if t is not None and t < tbest:
                tbest = t
                best = (np.array([cx, cy], float), r + pad, t)
        return best

    def obstacle_tangential(self, p: np.ndarray, goal_dir: np.ndarray) -> np.ndarray:
        """
        Engelin etrafında 'dolaştıran' akış: normal repulsif kuvvete ek olarak
        goal yönüne uygun yönde teğetsel (perp) bileşen üretir.
        """
        g = goal_dir / (np.linalg.norm(goal_dir) + 1e-8)
        acc = np.zeros(2)
        for cx, cy, r in self.cfg.OBST:
            c = np.array([cx, cy], float)
            v = p - c
            dist = np.linalg.norm(v) + 1e-8
            R = r + self.cfg.OBS_INFLUENCE
            if dist < R:
                n = v / dist
                # iki teğet yön: n⊥ = [+(-n_y, n_x)] ve [-( -n_y, n_x )]
                t1 = np.array([-n[1], n[0]])
                t2 = -t1
                # hedefe en uygun teğet (g ile daha paralel olan)
                if np.dot(t1, g) >= np.dot(t2, g):
                    t = t1
                else:
                    t = t2
                # yakınken güçlü, uzakta zayıf: ~ (1 / (dist - r + ε))
                gain = self.cfg.TANG_GAIN / ((dist - r + 0.5)**1.2)
                acc += gain * t
        # aşırı büyümeyi engelle
        n = np.linalg.norm(acc)
        if n > self.cfg.OBS_FORCE_MAX:
            acc = acc / (n + 1e-8) * self.cfg.OBS_FORCE_MAX
        return acc
    
# ────────────────────────────────────────────────────────────────────────────
# PathPlanner.py
# ────────────────────────────────────────────────────────────────────────────
class PathPlanner:
    """
    Şişirilmiş dairesel engeller için basit visibility-graph yol planlayıcı.
    - Her engelin etrafında N örnek nokta üretir (r+infl).
    - İki nokta arasına LOS (line-of-sight) varsa kenar ekler.
    - Dijkstra ile en kısa yol, ardından LOS-smoothing uygular.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.N_SAMPLES = 16  # engel başına örnek nokta
        self.INFL = max(0.4, getattr(cfg, "SAFE_PAD", 0.6) + getattr(cfg, "BYPASS_MARGIN", 0.8) - 0.2)

    @staticmethod
    def _unit(v):
        n = np.linalg.norm(v) + 1e-8
        return v / n

    @staticmethod
    def _seg_circle_hit(p, q, c, r):
        # [p,q] parçası C merkezli r çemberini kesiyor mu?
        pq = q - p
        t = np.dot(c - p, pq) / (np.dot(pq, pq) + 1e-12)
        t = np.clip(t, 0.0, 1.0)
        closest = p + t * pq
        return np.linalg.norm(closest - c) <= r

    def _in_world(self, p):
        x0, x1, y0, y1 = self.cfg.WORLD
        pad = getattr(self.cfg, "SAFE_PAD", 0.6)
        return (x0+pad <= p[0] <= x1-pad) and (y0+pad <= p[1] <= y1-pad)

    def _los(self, p, q):
        # Dünya içi ve engel çakışması yoksa true
        if (not self._in_world(p)) or (not self._in_world(q)):
            return False
        for cx, cy, r in self.cfg.OBST:
            c = np.array([cx, cy], float)
            R = r + self.INFL
            if self._seg_circle_hit(p, q, c, R):
                return False
        return True

    def _sample_nodes(self, start, goal):
        nodes = [start, goal]
        for cx, cy, r in self.cfg.OBST:
            angles = np.linspace(0, 2*np.pi, self.N_SAMPLES, endpoint=False)
            R = r + self.INFL
            pts = np.c_[R*np.cos(angles)+cx, R*np.sin(angles)+cy]
            for p in pts:
                if self._in_world(p):
                    nodes.append(p)
        return np.array(nodes)

    def _graph(self, nodes):
        n = len(nodes)
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if self._los(nodes[i], nodes[j]):
                    w = float(np.linalg.norm(nodes[i]-nodes[j]))
                    adj[i].append((j, w))
                    adj[j].append((i, w))
        return adj

    @staticmethod
    def _dijkstra(adj, s, t):
        n = len(adj)
        dist = [1e18]*n; prev = [-1]*n
        dist[s] = 0.0
        pq = [(0.0, s)]
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist[u]: continue
            if u == t: break
            for v, w in adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if dist[t] >= 1e17:
            return None
        path = []
        cur = t
        while cur != -1:
            path.append(cur)
            cur = prev[cur]
        return path[::-1]

    def _smooth(self, nodes, path_idx):
        if path_idx is None:
            return []
        pts = [nodes[i] for i in path_idx]
        i = 0
        out = [pts[0]]
        while i < len(pts)-1:
            j = len(pts)-1
            while j > i+1 and (not self._los(pts[i], pts[j])):
                j -= 1
            out.append(pts[j])
            i = j
        return out

    def plan(self, start, goal):
        nodes = self._sample_nodes(start, goal)
        adj = self._graph(nodes)
        s, t = 0, 1  # start, goal
        path_idx = self._dijkstra(adj, s, t)
        if path_idx is None:
            return [goal]  # fallback

        # 1) mevcut smoothing
        smooth_pts = self._smooth(nodes, path_idx)

        # 2) EKLE: shortcut smoothing (kısayol denemeleri)
        smooth_pts = self._shortcut(smooth_pts, iters=16)

        return smooth_pts if len(smooth_pts) >= 2 else [goal]

    
    def _shortcut(self, pts, iters=12):
        """Basit iki-nokta kısayol: A..B arasını tek segmentle değiş, LOS varsa."""
        if len(pts) <= 2: return pts
        pts = pts[:]
        n = len(pts)
        for _ in range(iters):
            i = np.random.randint(0, n-2)
            j = np.random.randint(i+2, n)
            if self._los(pts[i], pts[j]):
                pts = pts[:i+1] + pts[j:]
                n = len(pts)
                if n <= 2: break
        return pts


# ──────────────────────────────────────────────────────────────────────────────
# control.py (SwarmController)
# ──────────────────────────────────────────────────────────────────────────────
class SwarmController:
    def __init__(self, cfg: Config, env: Environment, drones: List[Drone], leader_idx: int = 0):
        self.cfg = cfg
        self.env = env
        self.drones = drones
        self.leader = leader_idx

        # Emniyetli formasyon yarıçapı
        r_min = (getattr(self.cfg, "D_MIN", 1.2) * 1.1) / (2 * math.sin(math.pi / self.cfg.N))
        self.R_FORM = max(1.5, r_min)
        angles = np.linspace(0, 2*np.pi, cfg.N, endpoint=False)
        self.OFFSETS = np.c_[self.R_FORM*np.cos(angles), self.R_FORM*np.sin(angles)]

        self.assignments = np.arange(cfg.N)
        self.frame_count = 0
        self.mode = "waypoint"
        self.sel_obs = 0
        self.wp_idx = 0

        # Global yol planlayıcı ve rota kuyruğu
        self.planner = PathPlanner(cfg)
        self.path: List[np.ndarray] = []
        self.path_i = 0
        self.REPLAN_PERIOD = 45
        self._last_plan_frame = -999

        self.slot_lp: List[np.ndarray] = [np.zeros(2) for _ in range(self.cfg.N)]

        # Pure-Pursuit ayarları
        self.LOOKAHEAD_M = getattr(cfg, "LOOKAHEAD_M", 3.2)  # temel bakış mesafesi (m)
        self.LOOKAHEAD_K = getattr(cfg, "LOOKAHEAD_K", 0.9)  # hıza bağlı ek (m per m/s)

    # ---- helpers ----
    def positions(self) -> np.ndarray:
        return np.array([d.pos for d in self.drones])

    def velocities(self) -> np.ndarray:
        return np.array([d.vel for d in self.drones])

    def neighbors(self, i: int, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d = pos - pos[i]
        dist = np.linalg.norm(d, axis=1)
        idx = (dist < self.cfg.R_NEIGH) & (dist > 1e-8)
        vel = self.velocities()
        return d[idx], vel[idx]

    def boids_terms(self, i: int, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        d, vN = self.neighbors(i, pos)
        if d.shape[0] == 0:
            return np.zeros(2), np.zeros(2), np.zeros(2)
        dist = np.linalg.norm(d, axis=1)
        mask = (dist < getattr(self.cfg, "R_SEP", 1.6))
        if np.any(mask):
            dm = dist[mask][:, None]
            sep = (-d[mask] / (dm**2)).sum(axis=0)
        else:
            sep = np.zeros(2)
        ali = vN.mean(axis=0) - self.drones[i].vel
        neigh_pos = pos[i] + d
        coh = neigh_pos.mean(axis=0) - pos[i]
        return sep, ali, coh

    def current_goal(self) -> np.ndarray:
        if self.mode == "waypoint":
            return self.cfg.WAYPOINTS[self.wp_idx]
        else:
            return self.cfg.OBST[self.sel_obs, :2]

    def _update_offsets(self, pos):
        gap = self._nearest_obstacle_gap(pos[self.leader])
        obsR = getattr(self.cfg, "OBS_INFLUENCE", 5.0)
        R_min = (self.cfg.D_MIN*1.1) / (2*math.sin(math.pi/self.cfg.N))
        R_max = getattr(self.cfg, "R_FORM_MAX", self.R_FORM)
        t = np.clip(gap / max(obsR, 1e-6), 0.0, 1.0)           # 0: çok dar, 1: ferah
        R_des = R_min + (R_max - R_min) * t
        a = getattr(self.cfg, "R_FORM_ADAPT_ALPHA", 0.12)
        self.R_FORM = (1-a)*self.R_FORM + a*R_des              # low-pass
        ang = np.linspace(0, 2*np.pi, self.cfg.N, endpoint=False)
        self.OFFSETS = np.c_[self.R_FORM*np.cos(ang), self.R_FORM*np.sin(ang)]


    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v) + 1e-8
        return v / n

    def leader_anchor(self, pos, vel):
        # İstersen ADAPTIVE_LEAD kullan; burada basit lead
        tau = getattr(self.cfg, "LEAD_AHEAD_TAU", 0.4)
        return pos[self.leader] + vel[self.leader] * tau

    def _nearest_obstacle_gap(self, p: np.ndarray) -> float:
        gap = 1e9
        for cx, cy, r in self.cfg.OBST:
            gap = min(gap, np.linalg.norm(p - np.array([cx, cy])) - r)
        return gap

    # ---- planlama ----
    def _replan_if_needed(self, pos):
        # Rota yoksa, eskidiyse veya sıradaki segment görünür değilse yeniden planla
        need = (self.frame_count - self._last_plan_frame >= self.REPLAN_PERIOD) or (len(self.path) == 0)
        if not need and self.path_i < len(self.path):
            P = pos[self.leader]
            Q = self.path[self.path_i]
            if self.planner._los(P, Q):
                return
            need = True
        if need:
            start = pos[self.leader]
            goal = self.current_goal()
            self.path = self.planner.plan(start, goal)
            self.path_i = 0
            self._last_plan_frame = self.frame_count

    def _path_target(self, pos: np.ndarray, vel: np.ndarray, base_goal: np.ndarray) -> np.ndarray:
        """
        Pure-Pursuit: liderin pozisyonundan itibaren rotada LOOKAHEAD kadar ilerideki noktayı hedef seç.
        Yol biterse base_goal'e dön.
        """
        # Dinamik lookahead: hızlıyken daha ileriye bak
        Ld = float(self.LOOKAHEAD_M + self.LOOKAHEAD_K * np.linalg.norm(vel[self.leader]))
        if self.path_i >= len(self.path):
            return base_goal

        P = pos[self.leader]
        pts = [P] + [self.path[k] for k in range(self.path_i, len(self.path))]
        # Ardışık segmentlerde kümülatif mesafe
        accum = 0.0
        for k in range(len(pts) - 1):
            a, b = pts[k], pts[k+1]
            seg = np.linalg.norm(b - a)
            if accum + seg >= Ld:
                t = (Ld - accum) / max(seg, 1e-8)
                return a + t * (b - a)
            accum += seg
        # Rota kısa ise son noktayı dön
        return pts[-1]

    # ---- assignment ----
    def recompute_assignments(self, pos: np.ndarray) -> None:
        vel = self.velocities()
        anchor = self.leader_anchor(pos, vel)
        desired_slots = anchor + self.OFFSETS
        cost = np.linalg.norm(pos[:, None, :] - desired_slots[None, :, :], axis=2)
        for i in range(self.cfg.N):
            for j in range(self.cfg.N):
                if j != self.assignments[i]:
                    cost[i, j] += self.cfg.SWITCH_PENALTY
        try:
            if _SCIPY_OK:
                row_ind, col_ind = linear_sum_assignment(cost)
                self.assignments = col_ind
            else:
                self.assignments = self._greedy_assign(cost)
        except Exception as e:
            logging.warning(f"Assignment failed, fallback identity. Err={e}")
            self.assignments = np.arange(self.cfg.N)

    @staticmethod
    def _greedy_assign(cost: np.ndarray) -> np.ndarray:
        N = cost.shape[0]
        assigned = -np.ones(N, dtype=int)
        rows = set(range(N)); cols = set(range(N))
        while rows:
            i = min(rows)
            j = min(cols, key=lambda c: cost[i, c])
            assigned[i] = j
            rows.remove(i); cols.remove(j)
        return assigned

    def _limit_turn(self, v_curr: np.ndarray, v_des: np.ndarray) -> np.ndarray:
        limit = getattr(self.cfg, "TURN_RATE_MAX", 0.25)
        s_curr = np.linalg.norm(v_curr)
        s_des  = np.linalg.norm(v_des)
        if s_des < 1e-6:
            return v_des
        u_curr = v_curr / (s_curr + 1e-8) if s_curr > 1e-6 else v_des / s_des
        u_des  = v_des / s_des
        dot = np.clip(np.dot(u_curr, u_des), -1.0, 1.0)
        ang = math.acos(dot)
        if ang <= limit:
            return v_des
        sign = 1.0 if np.cross(np.r_[u_curr,0], np.r_[u_des,0])[2] >= 0 else -1.0
        cosA = math.cos(limit); sinA = math.sin(limit)
        R = np.array([[cosA, -sign*sinA],[sign*sinA, cosA]])
        u_new = R.dot(u_curr)
        return u_new * s_des

    # ---- main step ----
    def step(self) -> None:
        cfg = self.cfg
        pos = self.positions()
        vel = self.velocities()
        self._update_offsets(pos)

        # Waypoint ilerlet (yalnızca ana hedefe varınca)
        if self.mode == "waypoint":
            if np.linalg.norm(pos[self.leader] - self.cfg.WAYPOINTS[self.wp_idx]) < 1.5:
                self.wp_idx = (self.wp_idx + 1) % len(self.cfg.WAYPOINTS)
                self.path = []
                self.path_i = 0

        # Plan/ replan
        self._replan_if_needed(pos)

        # Lider hedefi: Pure-Pursuit lookahead
        base_goal = self.current_goal() if self.path_i >= len(self.path) else self.path[self.path_i]
        leader_goal = self._path_target(pos, vel, base_goal)
        if self.path_i < len(self.path) and np.linalg.norm(self.path[self.path_i] - pos[self.leader]) < 1.0:
            self.path_i += 1  # sıradaki ara hedefe geç

        # Periyodik yeniden atama (dar alanda dondur)
        if self.frame_count % cfg.ASSIGN_PERIOD == 0:
            gap = self._nearest_obstacle_gap(pos[self.leader])
            if gap > getattr(cfg, "ASSIGN_FREEZE_GAP", 2.0):  # genişse yeniden ata
                self.recompute_assignments(pos)
        self.frame_count += 1

        # Grup hız tavanı: en kötü drone yerine YÜZDELİK
        leader_dir = self._unit(leader_goal - pos[self.leader]) if np.linalg.norm(leader_goal - pos[self.leader]) > 1e-6 else self._unit(vel[self.leader])
        pad = getattr(self.cfg, "SAFE_PAD", 0.6)
        ttc_slow = getattr(self.cfg, "TTC_SLOW", 2.5)
        perc = float(getattr(self.cfg, "GROUP_CLEAR_PERCENTILE", 30.0))  # 0..100
        clear_list = []
        for i in range(cfg.N):
            if hasattr(self.env, "forward_clearance"):
                cl = self.env.forward_clearance(pos[i], leader_dir, pad)
            else:
                # kaba yaklaşık: LOS varsa büyük, yoksa küçük bir değer
                cl = 1000.0 if self.planner._los(pos[i], pos[i] + leader_dir * 1000.0) else 3.0
            clear_list.append(cl)
        cl_soft = float(np.percentile(np.array(clear_list), perc))
        v_cap_group = min(cfg.V_MAX_FAR, cl_soft / max(ttc_slow, 1e-6))
        
        anchor = self.leader_anchor(pos, vel)
        max_lag = 0.0
        for j in range(cfg.N):
            if j == self.leader: 
                continue
            slotj = self.assignments[j]
            desired_j = anchor + self.OFFSETS[slotj]
            lag = np.linalg.norm(desired_j - pos[j])
            if lag > max_lag:
                max_lag = lag

        # 0..1 arası ölçek: hata küçükken 1, büyüyünce düşsün
        beta = getattr(cfg, "COHESION_BETA", 0.10)   # hassasiyet
        lag_clip = min(max_lag, 3.0)                 # 3m üstünü doygun say
        scale = float(np.clip(1.0 - beta * lag_clip, 0.6, 1.0))

        v_cap_group *= scale

        # Lider engelden uzaktaysa grup tavanını kaldır (daha atak sürüş)
        gap_leader = self._nearest_obstacle_gap(pos[self.leader])
        if gap_leader > getattr(cfg, "OBS_INFLUENCE", 5.0) + 0.5:
            v_cap_group = cfg.V_MAX_FAR

        accel = np.zeros_like(pos)

        for i in range(cfg.N):
            sep, ali, coh = self.boids_terms(i, pos)

            # Yakın çarpışma güçlendirme (ajan-ajan)
            d_all = np.linalg.norm(pos - pos[i], axis=1)
            if np.any((d_all > 1e-8) & (d_all < cfg.D_MIN)):
                sep *= 2.0

            # Potansiyel alanlar: duvar + engel
            obs = self.env.wall_repulsion(pos[i]) + self.env.obstacle_repulsion(pos[i])

            # Liderin engel itişini "acil durum" dışında zayıflat (planlı yol öncelikli)
            if i == self.leader:
                if gap_leader > (pad + 0.6 * getattr(self.cfg, "OBS_INFLUENCE", 5.0)):
                    obs *= 0.25  # uzakken zayıf itiş

            # Takipçiler: uzaktayken engel itişini kıs (formasyon bozulmasın)
            if i != self.leader:
                near = self._nearest_obstacle_gap(pos[i])
                if near > (pad + 0.6 * getattr(self.cfg, "OBS_INFLUENCE", 5.0)):
                    obs *= 0.25

            if i == self.leader:
                vec = leader_goal - pos[i]
                dist = np.linalg.norm(vec)
                v_des = U.speed_profile(dist, cfg.V_MIN_NEAR, cfg.V_MAX_FAR, cfg.SLOW_RADIUS)
                v_des = min(v_des, v_cap_group)
                v_goal = v_des * (self._unit(vec) if dist > 1e-6 else np.zeros(2))
                v_goal = self._limit_turn(vel[i], v_goal)
                acc_goal = U.accel_to_target_velocity(vel[i], v_goal, cfg.K_VEL_TRACK)
                acc_form = np.zeros(2)
            else:
                # follower target (anchor + offset)
                slot = self.assignments[i]
                anchor = self.leader_anchor(pos, vel)
                raw_desired_pos = anchor + self.OFFSETS[slot]

                # slot hedefini yumuşat (LPF): jitter azalır → RMSE düşer
                a = getattr(cfg, "SLOT_LPF_ALPHA", 0.25)
                self.slot_lp[i] = (1 - a) * self.slot_lp[i] + a * raw_desired_pos
                desired_pos = self.slot_lp[i]

                vec = desired_pos - pos[i]
                dist = np.linalg.norm(vec)

                # hız tavanı (grup) uygulaması aynen kalsın
                v_des = U.speed_profile(dist, cfg.V_MIN_NEAR, cfg.V_MAX_FAR, cfg.SLOW_RADIUS)
                v_des = min(v_des, v_cap_group)
                v_slot = v_des * (self._unit(vec) if dist > 1e-6 else np.zeros(2))

                # ---- ADAPTİF PD KAZANÇLARI ----
                # dar alanda (gap küçük) P/D biraz düşür; ferah alanda yükselt
                gap = self._nearest_obstacle_gap(pos[self.leader])
                gmax = getattr(self.cfg, "OBS_INFLUENCE", 5.0)
                ratio = float(np.clip(gap / max(gmax, 1e-6), 0.0, 1.0))  # 0=dar, 1=ferah

                Kp_min, Kp_max = 1.6, max(getattr(cfg, "K_FORM_P", 2.2), 2.2)
                Kd_min, Kd_max = 0.9, max(getattr(cfg, "K_FORM_D", 1.1), 1.1)
                Kp = Kp_min + (Kp_max - Kp_min) * ratio
                Kd = Kd_min + (Kd_max - Kd_min) * ratio

                pos_err = vec
                vel_err = v_slot - vel[i]

                acc_goal = np.zeros(2)  # takipçi için yok
                acc_form = Kp * pos_err + Kd * vel_err

            u = (cfg.W_SEP*sep + cfg.W_ALI*ali + cfg.W_COH*coh + cfg.W_OBS*obs) \
                + (cfg.W_GOAL*acc_goal) + (cfg.W_FORM*acc_form)

            a = U.unit(u) * min(np.linalg.norm(u), cfg.ACC_MAX)
            accel[i] = a

        # Entegrasyon ve çarpışma çözümü
        vel = (1 - cfg.DAMPING)*vel + accel*cfg.DT
        vel = U.limit_speed(vel, cfg.MAX_SPEED)
        pos = pos + vel*cfg.DT

        for i in range(cfg.N):
            pos[i], vel[i] = self.env.resolve_collisions(pos[i], vel[i])

        for i, d in enumerate(self.drones):
            d.vel = vel[i]
            prev = d.pos.copy()
            d.pos = pos[i]
            d.add_path(prev)



# ──────────────────────────────────────────────────────────────────────────────
# metrics.py
# ──────────────────────────────────────────────────────────────────────────────
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
            if i == leader: continue
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

# ──────────────────────────────────────────────────────────────────────────────
# viz.py (Matplotlib Visualizer)
# ──────────────────────────────────────────────────────────────────────────────
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
    
# ──────────────────────────────────────────────────────────────────────────────
# mavlink_bridge.py
# ─────────────────────────────────────────────────────────────────────────────
class MAVLinkBridge:
    def __init__(self, url="udpout:127.0.0.1:14550", sysid=1, compid=1):
        self.m = mavutil.mavlink_connection(url, source_system=sysid, source_component=compid)

    def start(self):
        while True:
            self.m.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_QUADROTOR,
                mavutil.mavlink.MAV_AUTOPILOT_PX4,
                0, 0, mavutil.mavlink.MAV_STATE_ACTIVE
            )
            time.sleep(1)

# ──────────────────────────────────────────────────────────────────────────────
# odompub.py
# ─────────────────────────────────────────────────────────────────────────────
class OdomPub(Node):
    def __init__(self):
        super().__init__('odom_pub')
        self.pub = self.create_publisher(Odometry, 'uav0/odom', 10)

    def tick(self, state):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = state.pos
        self.pub.publish(msg)

# ──────────────────────────────────────────────────────────────────────────────
# gps.py
# ──────────────────────────────────────────────────────────────────────────────
class GPS:
    def __init__(self, sigma=1.5, bias_walk=0.01, dropout_p=0.02, seed=0):
        self.rng = np.random.default_rng(seed)
        self.bias = np.zeros(3)
        self.sigma = sigma; self.bias_walk = bias_walk; self.dropout_p = dropout_p

    def read(self, true_pos):
        if self.rng.random() < self.dropout_p:
            return None  # kayıp
        self.bias += self.rng.normal(0, self.bias_walk, 3)
        noise = self.rng.normal(0, self.sigma, 3)
        return true_pos + self.bias + noise

# ──────────────────────────────────────────────────────────────────────────────
# a_star.py
# ─────────────────────────────────────────────────────────────────────────────
def astar(neigh_fn, h_fn, start, goal):
    openq=[(0,start)]; g={start:0}; parent={start:None}
    while openq:
        _, u = heapq.heappop(openq)
        if u==goal: break
        for v, w in neigh_fn(u):
            alt = g[u] + w
            if v not in g or alt < g[v]:
                g[v]=alt; parent[v]=u
                f = alt + h_fn(v, goal)
                heapq.heappush(openq, (f, v))
    # rekonstrüksiyon
    path=[]; x=goal
    while x is not None: path.append(x); x=parent.get(x)
    return list(reversed(path)), g.get(goal, math.inf)

# ──────────────────────────────────────────────────────────────────────────────
# test_channel.py
# ─────────────────────────────────────────────────────────────────────────────
def test_loss_rate():
    ch=LossyChannel(loss_p=0.1, jitter=0.0, base_delay=0.0)
    n=2000; sent=0; recv=0
    for i in range(n):
        ch.send(i); sent+=1
    import time; time.sleep(0.1)
    recv=len(ch.recv_ready())
    loss=(sent-recv)/sent
    assert 0.05 < loss < 0.15


# ─────────────────────────────────────────────────────────────────────────────
# fastapi.py
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI()
clients = set()

@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept(); clients.add(ws)
    try:
        while True:
            await ws.receive_text()  # ping/pong
    finally:
        clients.remove(ws)

async def push_state(state_dict):
    msg = json.dumps(state_dict, default=float)
    for c in list(clients):
        await c.send_text(msg)    

# ─────────────────────────────────────────────────────────────────────────────
# main.py
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO)
    np.random.seed(2)

    cfg = Config()

    # init drones
    drones: List[Drone] = []
    for i in range(cfg.N):
        pos = np.array([np.random.uniform(2, 8), np.random.uniform(20, 24)], dtype=float)
        vel = np.random.randn(2) * 0.1
        drones.append(Drone(i, pos, vel))

    env = Environment(cfg)
    swarm = SwarmController(cfg, env, drones, leader_idx=0)
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

