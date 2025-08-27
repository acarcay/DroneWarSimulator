import math
import heapq
import numpy as np

from config import Config


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
            if d != dist[u]:
                continue
            if u == t:
                break
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
        if len(pts) <= 2:
            return pts
        pts = pts[:]
        n = len(pts)
        for _ in range(iters):
            i = np.random.randint(0, n-2)
            j = np.random.randint(i+2, n)
            if self._los(pts[i], pts[j]):
                pts = pts[:i+1] + pts[j:]
                n = len(pts)
                if n <= 2:
                    break
        return pts
