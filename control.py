import logging
import math
from typing import List, Tuple

import numpy as np

from config import Config
from environment import Environment, PathPlanner
from models import Drone
from utils import U

try:
    from scipy.optimize import linear_sum_assignment
    _SCIPY_OK = True
except Exception:  # pragma: no cover - SciPy optional
    _SCIPY_OK = False


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
        pts = self.path[self.path_i:]
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
        rows = set(range(N))
        cols = set(range(N))
        while rows:
            i = min(rows)
            j = min(cols, key=lambda c: cost[i, c])
            assigned[i] = j
            rows.remove(i)
            cols.remove(j)
        return assigned

    def _limit_turn(self, v_curr: np.ndarray, v_des: np.ndarray) -> np.ndarray:
        limit = getattr(self.cfg, "TURN_RATE_MAX", 0.25)
        s_curr = np.linalg.norm(v_curr)
        s_des = np.linalg.norm(v_des)
        if s_des < 1e-6:
            return v_des
        u_curr = v_curr / (s_curr + 1e-8) if s_curr > 1e-6 else v_des / s_des
        u_des = v_des / s_des
        dot = np.clip(np.dot(u_curr, u_des), -1.0, 1.0)
        ang = math.acos(dot)
        if ang <= limit:
            return v_des
        sign = 1.0 if np.cross(np.r_[u_curr, 0], np.r_[u_des, 0])[2] >= 0 else -1.0
        cosA = math.cos(limit)
        sinA = math.sin(limit)
        R = np.array([[cosA, -sign*sinA], [sign*sinA, cosA]])
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

