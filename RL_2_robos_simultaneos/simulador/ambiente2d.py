import numpy as np

class Ambiente2D:
    def __init__(self,
                 world_bounds=5.0,
                 door_width=1.0,
                 wall_y=0.0,
                 dt=0.02,
                 robot_radius=0.25):

        self.world_bounds = world_bounds
        self.door_width = door_width
        self.wall_y = wall_y
        self.dt = dt
        self.robot_radius = robot_radius

        self.door_x_min = -door_width/2.0
        self.door_x_max =  door_width/2.0

        self.start1 = np.array([-0.4, -4.0, np.pi/2])
        self.start2 = np.array([ 0.4, -4.0, np.pi/2])

        self.destino = np.array([0.0, 3.0])
        self.reset()

    # ===========================================================
    def reset(self):
        self.pose1 = self.start1.copy()
        self.pose2 = self.start2.copy()

        self.passed1 = False
        self.passed2 = False
        self.pass_time1 = None
        self.pass_time2 = None

        self.prev_d1 = None
        self.prev_d2 = None
        self.t = 0

        s1, d1 = self._compute_state_from_pose(self.pose1)
        s2, d2 = self._compute_state_from_pose(self.pose2)

        self.prev_d1 = d1
        self.prev_d2 = d2

        return np.concatenate([s1, s2]).astype(np.float32)

    # ===========================================================
    def _compute_state_from_pose(self, pose):
        dx = self.destino[0] - pose[0]
        dy = self.destino[1] - pose[1]
        dist = np.sqrt(dx*dx + dy*dy)

        if dist < 1e-6:
            nx = ny = 0.0
        else:
            nx = dx / dist
            ny = dy / dist

        angle_to_goal = np.arctan2(dy, dx)
        rel_ang = np.sin(angle_to_goal - pose[2])   # mede o erro angular entre a orientação do robô e a direção do destino.

        max_dist = np.sqrt((2*self.world_bounds)**2 + (2*self.world_bounds)**2)
        dist_norm = dist / max_dist

        return np.array([nx, ny, rel_ang, dist_norm], dtype=np.float32), dist

    # ===========================================================
    def _step_robot(self, pose, control):
        v, w = control
        new_pose = pose.copy()

        nsteps = 5
        dt_i = self.dt / nsteps

        for _ in range(nsteps):
            theta = new_pose[2]

            new_pose[0] += dt_i * v * np.cos(theta)
            new_pose[1] += dt_i * v * np.sin(theta)
            new_pose[2] = (new_pose[2] + dt_i * w) % (2*np.pi)

        return new_pose

    # ===========================================================
    def _check_wall_collision(self, pose):
        x, y = pose[0], pose[1]

        if abs(y - self.wall_y) <= self.robot_radius:
            if not (self.door_x_min + self.robot_radius <= x <= self.door_x_max - self.robot_radius):
                return True


        if abs(x) > self.world_bounds or abs(y) > self.world_bounds:
            return True

        return False

    def _check_robot_collision(self, pose1, pose2):
        return np.linalg.norm(pose1[:2] - pose2[:2]) < (1.5 * self.robot_radius)

    # ===========================================================
    def step(self, action):
        a1, a2 = action
        self.pose1 = self._step_robot(self.pose1, a1)  # líder
        self.pose2 = self._step_robot(self.pose2, a2)  # seguidor

        done = False
        reward = 0.0

        # colisões
        if self._check_wall_collision(self.pose1) or \
        self._check_wall_collision(self.pose2) or \
        self._check_robot_collision(self.pose1, self.pose2):

            s1, _ = self._compute_state_from_pose(self.pose1)
            s2, _ = self._compute_state_from_pose(self.pose2)
            obs = np.concatenate([s1, s2]).astype(np.float32)

            return obs, -10.0, True, {}

        # estados
        prev_d1 = self.prev_d1
        prev_d2 = self.prev_d2

        s1, d1 = self._compute_state_from_pose(self.pose1)
        s2, d2 = self._compute_state_from_pose(self.pose2)

        # calcular distância entre robôs
        dist_between = np.linalg.norm(self.pose1[:2] - self.pose2[:2])

        # recompensa por se mover na direção certa (líder e seguidor)
        reward += (prev_d1 - d1) + (prev_d2 - d2)

        # recompensa por alinhamento
        reward += 0.1 * (1.0 - abs(s1[2]))  # líder
        reward += 0.1 * (1.0 - abs(s2[2]))  # seguidor

        # --- esquema líder-seguidor ---
        dx = self.pose1[0] - self.pose2[0]   # diferença no eixo X
        dy = self.pose1[1] - self.pose2[1]   # diferença no eixo Y
        dist_ls = np.sqrt(dx*dx + dy*dy)

        # seguidor deve manter distância moderada do líder
        if 0.5 <= dist_ls <= 1.0:
            reward += 0.3   # bom seguidor
        elif dist_ls > 2.0:
            reward -= 0.3   # se afastou demais
        elif dist_ls < 0.3:
            reward -= 0.3   # ficou colado demais

        # penalidade por tempo
        reward -= 0.05

        # atualizar prev distances
        self.prev_d1 = d1
        self.prev_d2 = d2

        # passagem pela porta
        if (self.pose1[1] > self.wall_y) and \
        (self.door_x_min - 0.1 <= self.pose1[0] <= self.door_x_max + 0.1):
            self.passed1 = True
            self.pass_time1 = self.t
            reward += 50.0
            if self.pass_time2 is not None and abs(self.pass_time1 - self.pass_time2) <= 40:
                reward += 200.0

        if not self.passed2 and (self.pose2[1] > self.wall_y+0.1) and \
        (self.door_x_min <= self.pose2[0] <= self.door_x_max):
            self.passed2 = True
            self.pass_time2 = self.t
            reward += 50.0
            if self.pass_time1 is not None and self.pass_time2 is not None:
                delta = abs(self.pass_time1 - self.pass_time2)
                reward += max(0, 200 - 5*delta)

        self.t += 1
        obs = np.concatenate([s1, s2]).astype(np.float32)

        return obs, float(reward), bool(done), {}

    def render_state(self):
        return {
            'pose1': self.pose1.copy(),
            'pose2': self.pose2.copy(),
            'door_x_min': self.door_x_min,
            'door_x_max': self.door_x_max,
            'wall_y': self.wall_y,
            'destino': self.destino.copy()
        }
