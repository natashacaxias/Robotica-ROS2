import numpy as np
from collections import deque

class LeaderFollowerEnv:
    def __init__(self):
        # Fisica
        self.dt = 0.05
        self.sub = 4
        self.delay = 4
        self.buf_size = 200

        # Ações Golden Ratio (Fibonacci: 5 e 8)
        self.v_vals = np.linspace(0.0, 0.5, 5)
        self.w_vals = np.linspace(-1.0, 1.0, 8)
        self.na = len(self.v_vals) * len(self.w_vals)

        # Seguidor
        self.max_v_f = 0.45
        self.kp_d = 0.8 # Suavizado (1.0 -> 0.8)
        self.kp_a = 1.4

        # ---- GEOMETRIA DA SALA 8x8 ----
        self.room_xmin = 0.0
        self.room_xmax = 8.0
        self.room_ymin = 0.0
        self.room_ymax = 8.0

        # Parede interna horizontal em y=4, de x=0 ate x=6
        self.inner_wall_y = 4.0
        self.inner_wall_xmin = 0.0
        self.inner_wall_xmax = 6.0

        # Gap (brecha) de x=6 ate x=8 (lado direito)
        self.gap_xmin = 6.0
        self.gap_xmax = 8.0

        # Porta de saida na parede DIREITA, PERTO DA PAREDE INFERIOR
        self.door_ymin = 0.5
        self.door_ymax = 1.5

        # Zona de spawn (amarela) — ACIMA da parede interna
        self.spawn_xmin = 1.5
        self.spawn_xmax = 3.5
        self.spawn_ymin = 5.0
        self.spawn_ymax = 7.0

        # Goal — fora da sala, alinhado com a porta
        self.goal_xmin = 9.0
        self.goal_xmax = 10.0
        self.goal_ymin = 0.7
        self.goal_ymax = 1.3

        # Seguranca (P3-AT: ~0.5m largura)
        self.min_safe_dist = 0.80
        self.collision_dist = 0.50
        self.desired_dist = 0.90
        self.spawn_radius = 0.80

        # Estado
        self.goal = None
        self.leader = np.zeros(3)
        self.follower = np.zeros(3)
        self.buffer = deque(maxlen=self.buf_size)

        # Checkpoints
        self.passed_breach = False
        self.passed_door = False

        # Logging
        self.v_leader = 0.0
        self.w_leader = 0.0
        self.v_follower = 0.0
        self.w_follower = 0.0

    @staticmethod
    def wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _collides_inner_wall(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        wy = self.inner_wall_y
        radius = self.collision_dist / 2.0
        if (y1 - wy) * (y2 - wy) <= 0:
            if abs(y2 - y1) > 1e-9:
                t = (wy - y1) / (y2 - y1)
                if 0 <= t <= 1:
                    xcross = x1 + t * (x2 - x1)
                    # O gap é entre 6.0 e 8.0. Com raio de 25cm, centro deve estar > 6.25
                    if xcross <= self.inner_wall_xmax + radius:
                        return True
        return False

    def _collides_room_walls(self, x, y):
        radius = self.collision_dist / 2.0
        # Paredes externas esquerda, baixo, cima
        if x < self.room_xmin + radius or y < self.room_ymin + radius or y > self.room_ymax - radius:
            return True
        # Parede direita com porta
        if x > self.room_xmax - radius:
            # So passa se o centro estiver entre door_min+pos e door_max-pos
            if self.door_ymin + radius <= y <= self.door_ymax - radius:
                return False
            return True
        return False

    def _is_through_breach(self, x, y):
        return (y < self.inner_wall_y and self.gap_xmin <= x <= self.gap_xmax)

    def _is_through_door(self, x, y):
        return (x >= self.room_xmax and self.door_ymin <= y <= self.door_ymax)

    def _step_uni(self, pose, v, w):
        x, y, th = pose
        for _ in range(self.sub):
            nx = x + self.dt * v * np.cos(th)
            ny = y + self.dt * v * np.sin(th)
            nth = self.wrap(th + self.dt * w)
            if self._collides_inner_wall((x, y), (nx, ny)): return np.array([x, y, nth])
            if self._collides_room_walls(nx, ny): return np.array([x, y, nth])
            x, y, th = nx, ny, nth
        return np.array([x, y, th])

    def _foll_ctrl(self, pose, tgt):
        x, y, th = pose
        tx, ty = tgt
        dx, dy = tx - x, ty - y
        dist = np.hypot(dx, dy)
        ang = np.arctan2(dy, dx)
        err = self.wrap(ang - th)
        if dist < self.min_safe_dist:
            return 0.0, np.clip(self.kp_a * err, -1.0, 1.0)
        v = np.clip(self.kp_d * dist, 0, self.max_v_f)
        w = np.clip(self.kp_a * err, -1.5, 1.5)
        return v, w

    def _state(self):
        Lx, Ly, Lth = self.leader
        Fx, Fy, Fth = self.follower
        gx, gy = self.goal
        
        # Define o alvo atual baseado nos checkpoints
        if not self.passed_breach:
            tx, ty = 7.0, 4.0 # Centro do gap
        elif not self.passed_door:
            tx, ty = 8.0, 1.0 # Centro da porta
        else:
            tx, ty = gx, gy # Objetivo final

        dist_target = np.hypot(tx - Lx, ty - Ly)
        pair = np.hypot(Lx - Fx, Ly - Fy)
        heading = np.arctan2(ty - Ly, tx - Lx)
        ang = self.wrap(Lth - heading)
        
        return np.array([
            tx - Lx, ty - Ly, dist_target, pair,
            np.sin(Lth), np.cos(Lth), np.sin(Fth), np.cos(Fth),
            ang,
            1.0 if self.passed_breach else 0.0,
            1.0 if self.passed_door else 0.0
        ], dtype=np.float32)

    def reset(self):
        lx = np.random.uniform(self.spawn_xmin + 0.3, self.spawn_xmax - 0.3)
        ly = np.random.uniform(self.spawn_ymin + 0.3, self.spawn_ymax - 0.3)
        lth = np.random.uniform(-np.pi, np.pi)
        self.leader = np.array([lx, ly, lth])
        for _ in range(40):
            ang = np.random.uniform(0, 2 * np.pi)
            d = np.random.uniform(0.15, self.spawn_radius)
            fx, fy = lx + d * np.cos(ang), ly + d * np.sin(ang)
            if self.room_xmin < fx < self.room_xmax and self.room_ymin < fy < self.room_ymax: break
        self.follower = np.array([fx, fy, np.random.uniform(-np.pi, np.pi)])
        self.goal = np.array([np.random.uniform(self.goal_xmin, self.goal_xmax),
                             np.random.uniform(self.goal_ymin, self.goal_ymax)])
        self.passed_breach = False
        self.passed_door = False
        self.buffer.clear()
        self.buffer.append(self.leader[:2].copy())
        return self._state()

    def step(self, action: int):
        vi = action // 8
        wi = action % 8
        v, w = self.v_vals[vi], self.w_vals[wi]
        self.leader = self._step_uni(self.leader, v, w)
        self.v_leader, self.w_leader = v, w
        self.buffer.append(self.leader[:2].copy())
        idx = max(0, len(self.buffer) - 1 - self.delay)
        vf, wf = self._foll_ctrl(self.follower, self.buffer[idx])
        self.follower = self._step_uni(self.follower, vf, wf)
        self.v_follower, self.w_follower = vf, wf

        st = self._state()
        dist_target = st[2]
        pair = st[3]
        ang = abs(st[8])

        # ---- RECOMPENSAS ----
        reward = -0.05
        
        # Alinhamento e Velocidade (Shaping)
        reward += 0.2 * self.v_leader * np.cos(ang)
        reward -= 0.1 * (self.w_leader**2)

        # Distância entre robôs (Manter estabilidade)
        dist_err = abs(pair - self.desired_dist)
        reward -= 2.0 * dist_err # Penaliza desvio dos 0.9m
        if dist_err < 0.1: reward += 0.5 # Bônus de "Safety Zone"

        # Colisões e Proximidade
        if pair < self.min_safe_dist: reward -= 20.0
        if pair < self.collision_dist: return st, -500.0, True, {"reason": "collision"}

        # Checkpoints e Sucesso
        if not self.passed_breach and self._is_through_breach(self.leader[0], self.leader[1]):
            self.passed_breach = True
            return self._state(), 300.0, False, {"checkpoint": "breach"}
            
        if self.passed_breach and not self.passed_door and self._is_through_door(self.leader[0], self.leader[1]):
            self.passed_door = True
            return self._state(), 500.0, False, {"checkpoint": "door"}

        # Magnet Jackpot (Fim)
        mx, my = self.goal
        dist_final = np.hypot(mx - self.leader[0], my - self.leader[1])
        if dist_final < 0.3 or (self.passed_door and self.leader[0] > self.room_xmax + 0.5):
            r_final = 1000.0
            if abs(self.leader[1] - my) < 0.2: r_final += 200
            return st, r_final, True, {"reason": "passed_goal"}

        return st, float(reward), False, {}
