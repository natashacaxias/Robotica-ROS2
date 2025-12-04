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
        self.buffer_size = 100  
        self.delay = 10          
        self.leader_buffer = []  
        self.arrival_hold_steps = 0
        self.arrival_hold_required = 18  # número de passos mantendo ambos próximos
        self.obj_pos = np.array([0.0, 3.0])
        self.obj_vel = np.zeros(2)
        self.v_max_obj = 1.0   # velocidade máxima do objetivo

        self.door_x_min = -door_width/2.0
        self.door_x_max =  door_width/2.0

        self.start1 = np.array([3, -4.0, 2*np.pi/3])   # líder
        self.start2 = np.array([3, -5, 2*np.pi/3])   # seguidor

        self.destino = np.array([0.0, 3.0])
        self.reset()

    def reset(self):
        self.pose1 = self.start1.copy() #pose atual do líder
        self.pose2 = self.start2.copy() #pose atual do seguidor

        self.passed1 = False
        self.passed2 = False
        self.pass_time1 = None
        self.pass_time2 = None

        self.prev_d1 = None
        self.prev_d2 = None
        self.t = 0
        self.arrival_hold_steps = 0

        s1, d1 = self._compute_state_from_pose(self.pose1)
        s2, d2 = self._compute_state_from_pose(self.pose2)

        self.prev_d1 = d1
        self.prev_d2 = d2

        self.leader_buffer = [self.pose1.copy()] * self.buffer_size # inicializa buffer com a pose inicial do líder

        # inicializa objetivo com velocidade aleatória
        speed = np.random.uniform(0, self.v_max_obj)
        angle = np.random.uniform(0, 2*np.pi)
        self.obj_vel = speed * np.array([np.cos(angle), np.sin(angle)])
        self.obj_pos = np.random.uniform(-2, 2, size=2)

        return np.concatenate([s1, s2]).astype(np.float32)

    def _compute_state_from_pose(self, pose, is_leader=True):
        if is_leader:
            destino = self.destino
        else:
            destino = getattr(self, "destino_seguidor", self.destino)

        dx = destino[0] - pose[0]
        dy = destino[1] - pose[1]

        dist = np.sqrt(dx*dx + dy*dy)

        if dist < 1e-6:
            nx = ny = 0.0
        else:
            nx = dx / dist # Para onde devo ir?
            ny = dy / dist

        angle_to_goal = np.arctan2(dy, dx)
        rel_ang = np.sin(angle_to_goal - pose[2])   # mede o erro angular entre a orientação do robô e a direção do destino. Estou apontando para o destino?

        max_dist = np.sqrt((2*self.world_bounds)**2 + (2*self.world_bounds)**2)
        dist_norm = dist / max_dist    # Quão longe estou?

        return np.array([nx, ny, rel_ang, dist_norm], dtype=np.float32), dist

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

    def _check_wall_collision(self, pose):
        x, y = pose[0], pose[1]

        if abs(y - self.wall_y) <= self.robot_radius:
            if (x - self.robot_radius) < self.door_x_min or (x + self.robot_radius) > self.door_x_max:
                return True

        if abs(x) > self.world_bounds or abs(y) > self.world_bounds:
            return True

        return False

    def _check_robot_collision(self, pose1, pose2):
        return np.linalg.norm(pose1[:2] - pose2[:2]) < (2 * self.robot_radius)

    def step(self, action):
        a1, a2 = action
        
        self.pose1 = self._step_robot(self.pose1, a1) # atualiza líder
        if self.pose1[1] < self.wall_y:
            self.destino = np.array([0.0, self.wall_y + 0.1])  # centro da porta
        else:
            self.destino = np.array([0.0, 3.0])  # destino final

        #if self._check_wall_collision(self.pose1):
            #print(f"💥 Colisão: líder bateu na parede em x={self.pose1[0]:.2f}, y={self.pose1[1]:.2f}")

        # atualizar objetivo móvel
        self.obj_pos += self.obj_vel * self.dt

        # líder persegue objetivo móvel
        self.destino = self.obj_pos.copy()

        self.leader_buffer.append(self.pose1.copy())
        if len(self.leader_buffer) > self.buffer_size:
            self.leader_buffer.pop(0)

        target_idx = max(0, len(self.leader_buffer) - 1 - self.delay)
        target_pose = self.leader_buffer[target_idx]
        self.destino_seguidor = target_pose[:2]  

        self.pose2 = self._step_robot(self.pose2, a2)

        done = False
        reward = 0.0

        # colisões
        if self._check_wall_collision(self.pose1) or \
        self._check_wall_collision(self.pose2) or \
        self._check_robot_collision(self.pose1, self.pose2):

            s1, _ = self._compute_state_from_pose(self.pose1)
            s2, _ = self._compute_state_from_pose(self.pose2)
            obs = np.concatenate([s1, s2]).astype(np.float32)

            return obs, -200.0, True, {}

        # estados
        prev_d1 = self.prev_d1
        prev_d2 = self.prev_d2

        s1, d1 = self._compute_state_from_pose(self.pose1, is_leader=True)
        s2, d2 = self._compute_state_from_pose(self.pose2, is_leader=False)

        # --- NOVO BLOCO: recompensa baseada em alvo móvel parametrizada ---
        v_obj = np.linalg.norm(self.obj_vel)  # velocidade linear do alvo
        dist_to_obj = np.linalg.norm(self.pose1[:2] - self.obj_pos)  # erro de distância do líder ao alvo

        d_opt = 0.3 + 0.7 * (v_obj / self.v_max_obj)  # distância ótima
        reward -= 10.0 * abs(dist_to_obj - d_opt)

        # --- INCENTIVO PARA PARAR QUANDO O ALVO ESTÁ PARADO ---
        if v_obj < 0.05:  # alvo praticamente parado
            v_lider = np.linalg.norm([a1[0]*np.cos(self.pose1[2]), a1[0]*np.sin(self.pose1[2])])
            
            # penaliza se o líder estiver se movendo
            reward -= 20.0 * v_lider

            # bônus se estiver bem próximo do alvo
            if dist_to_obj < 0.3:
                reward += 50.0


        dx = self.pose1[0] - self.pose2[0]  
        dy = self.pose1[1] - self.pose2[1]  
        dist_ls = np.sqrt(dx*dx + dy*dy)

        # 🚨 Encerrar episódio se robôs estiverem muito próximos
        if dist_ls < 0.2:
            reward -= 100
            done = True
            info = {"reason": "Colisão por proximidade extrema entre robôs"}

        if 0.3 <= dist_ls <= 1.0:
            reward += 50   # bom seguidor
        elif dist_ls > 2.0:
            reward -= 1   # se afastou demais
        elif dist_ls < 0.3:
            reward -= 100   # ficou colado demais

        # --- NOVO BLOCO: recompensa baseada em distância ótima do seguidor ---
        v_leader = np.linalg.norm([a1[0]*np.cos(self.pose1[2]), a1[0]*np.sin(self.pose1[2])])
        d_opt_seg = 0.3 + 0.7 * (v_leader / self.v_max_obj)
        reward -= 8.0 * abs(dist_ls - d_opt_seg)

        # penalidade por tempo
        reward -= 0.02

        # 🚨 penalidade por se aproximar da borda do mundo (líder)
        if abs(self.pose1[0]) > self.world_bounds - 0.5 or abs(self.pose1[1]) > self.world_bounds - 0.5:
            reward -= 50

        # 🚨 penalidade por se aproximar da borda do mundo (seguidor)
        if abs(self.pose2[0]) > self.world_bounds - 0.5 or abs(self.pose2[1]) > self.world_bounds - 0.5:
            reward -= 50

        # atualizar prev distances
        self.prev_d1 = d1
        self.prev_d2 = d2

        # --- PASSAGEM DO LÍDER ---
        if (not self.passed1) and (self.pose1[1] > self.wall_y) and \
            (self.door_x_min + self.robot_radius <= self.pose1[0] <= self.door_x_max - self.robot_radius):

            self.passed1 = True
            self.pass_time1 = self.t
            reward += 300.0  # bônus por passar primeiro

            # bônus por centralização
            centro_porta = (self.door_x_min + self.door_x_max) / 2
            dist_centro = abs(self.pose1[0] - centro_porta)
            reward += max(0, 150 - 100 * dist_centro)

        if self.pose1[1] > self.wall_y and not self.passed1:
            if not (self.door_x_min + self.robot_radius <= self.pose1[0] <= self.door_x_max - self.robot_radius):
                reward -= 100  # tentou atravessar fora da porta

        # --- PASSAGEM DO SEGUIDOR ---
        if (not self.passed2) and (self.pose2[1] > self.wall_y) and \
            (self.door_x_min + self.robot_radius <= self.pose2[0] <= self.door_x_max - self.robot_radius):

            self.passed2 = True
            self.pass_time2 = self.t
            reward += 200.0 

        # --- BÔNUS POR PASSAREM QUASE JUNTOS ---
        if self.passed1 and self.passed2 and not hasattr(self, "bonus_porta_dado"):
            delta = abs(self.pass_time1 - self.pass_time2)
            reward += max(0, 500 - 5 * delta)

            # marca que o bônus já foi dado
            self.bonus_porta_dado = True
              
        self.t += 1
        obs = np.concatenate([s1, s2]).astype(np.float32)


        # chegada ao destino 
        chegou1 = np.linalg.norm(self.pose1[:2] - self.destino) < 0.5
        chegou2 = np.linalg.norm(self.pose2[:2] - self.destino) < 0.5

        if chegou1 and chegou2:
            reward += 700.0
            dist_final = np.linalg.norm(self.pose1[:2] - self.pose2[:2])
            if dist_final < 0.5:
                reward += 100.0
            else:
                reward -= 300.0

            # manter próximos por N passos antes de encerrar (dwell)
            if dist_final < 0.5:
                self.arrival_hold_steps += 1
            else:
                self.arrival_hold_steps = 0

        else:
            # só um chegou
            if chegou1 or chegou2:
                reward += 100.0
            self.arrival_hold_steps = 0

        # penaliza movimento após chegada (incentiva desacelerar)
        if chegou1 or chegou2:
            reward -= 10 * (abs(a1[0]) + abs(a2[0]))

        # encerrar episódio quando ambos chegaram e ficaram próximos por alguns passos
        if self.arrival_hold_steps >= self.arrival_hold_required:
            done = True
            info = {"reason": "Ambos chegaram e permaneceram próximos no destino"}
        else:
            info = {}
        
        return obs, float(reward), bool(done), info

    def render_state(self):
        return {
            'pose1': self.pose1.copy(),
            'pose2': self.pose2.copy(),
            'door_x_min': self.door_x_min,
            'door_x_max': self.door_x_max,
            'wall_y': self.wall_y,
            'destino': self.destino.copy()
        }
