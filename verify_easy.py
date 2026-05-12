from evaluation.evaluator import _prepare_env_for_scenario, set_start_easy
from env.leader_follower_env import LeaderFollowerEnv
import numpy as np

env = LeaderFollowerEnv()
st = _prepare_env_for_scenario(env, set_start_easy)

print(f"Initial Leader: {env.leader}")
print(f"Goal: {env.goal}")

done = False
steps = 0
total_reward = 0
while not done and steps < 600:
    if steps % 2 == 0:
        action = 4 * 8 + 3 # 35
    else:
        action = 4 * 8 + 4 # 36
        
    st, r, done, info = env.step(action)
    total_reward += r
    steps += 1
    
    if steps % 100 == 0:
        print(f"Step {steps}: Leader at {env.leader[:2]}, dist_target={st[2]:.2f}, reward={r:.2f}")

    if done:
        print(f"Finished at step {steps}. Reason: {info.get('reason')}")
        print(f"Final Leader: {env.leader[:2]}")
        print(f"Final Reward: {total_reward:.2f}")
        break

if not done:
    print("Timeout")
