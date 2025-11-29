import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os

def animar_trajetoria(env, trajetorias, salvar_video_path=None, interval=40):
    """
    env: instancia do Ambiente2D
    trajetorias: lista de listas de states (cada step um state de 6 dims) OR
                 podemos obter pose via env.render_state during simulation
    salvar_video_path: full path to save .mp4 or None
    """
    # We'll query env.render_state() while stepping externally.
    fig, ax = plt.subplots(figsize=(6,6))
    world = env.world_bounds
    ax.set_xlim(-world-0.5, world+0.5)
    ax.set_ylim(-world-0.5, world+0.5)
    ax.set_aspect('equal')
    ax.set_title('Simulação 2D - Dois Robôs')

    # draw wall
    door_x_min = env.door_x_min
    door_x_max = env.door_x_max
    wall_y = env.wall_y

    # wall as two segments (left and right)
    left_wall = ax.plot([-env.world_bounds, door_x_min], [wall_y, wall_y], lw=6, solid_capstyle='butt')[0]
    right_wall = ax.plot([door_x_max, env.world_bounds], [wall_y, wall_y], lw=6, solid_capstyle='butt')[0]

    # destination
    dest = env.destino
    dest_marker = ax.plot(dest[0], dest[1], 'gx', markersize=12, label='Destino')[0]

    # robot artists
    rob1 = plt.Circle((0,0), env.robot_radius, fc='blue', ec='k')
    rob2 = plt.Circle((0,0), env.robot_radius, fc='red', ec='k')
    ax.add_patch(rob1)
    ax.add_patch(rob2)
    traj1_x, traj1_y = [], []
    traj2_x, traj2_y = [], []
    traj1_line, = ax.plot([], [], 'b--', linewidth=1)
    traj2_line, = ax.plot([], [], 'r--', linewidth=1)

    text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    frames = len(trajetorias)

    def init():
        traj1_line.set_data([], [])
        traj2_line.set_data([], [])
        text.set_text('')
        return rob1, rob2, traj1_line, traj2_line, text

    def update(i):
        info = trajetorias[i]  # aqui cada elemento é dict com poses ou state
        # support both dict (env.render_state style) or direct poses-state
        if isinstance(info, dict):
            p1 = info['pose1']
            p2 = info['pose2']
        else:
            # assume array-like [nx,ny,d1, nx2,ny2,d2] — can't draw orientation; use last saved poses
            p1 = info.get('pose1') if isinstance(info, dict) else None
            p2 = info.get('pose2') if isinstance(info, dict) else None
            if p1 is None or p2 is None:
                raise ValueError("trajetorias deve conter dicts vindos de env.render_state()")
        # update circles
        rob1.center = (p1[0], p1[1])
        rob2.center = (p2[0], p2[1])
        traj1_x.append(p1[0]); traj1_y.append(p1[1])
        traj2_x.append(p2[0]); traj2_y.append(p2[1])
        traj1_line.set_data(traj1_x, traj1_y)
        traj2_line.set_data(traj2_x, traj2_y)
        text.set_text(f"Step: {i}")
        return rob1, rob2, traj1_line, traj2_line, text

    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=interval)

    if salvar_video_path:
        # salvar com ffmpeg writer
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=1000//interval, metadata=dict(artist='me'), bitrate=2000)
        # ensure directory exists
        os.makedirs(os.path.dirname(salvar_video_path), exist_ok=True)
        ani.save(salvar_video_path, writer=writer)
        print(f"Vídeo salvo em: {salvar_video_path}")
    else:
        plt.show()
