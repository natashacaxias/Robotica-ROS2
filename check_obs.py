from env.leader_follower_env import LeaderFollowerEnv
import numpy as np

env = LeaderFollowerEnv()
obs = env.reset()
print(f"Obs shape: {obs.shape}")
print(f"Obs type: {type(obs)}")

st = env._state()
print(f"Single state len: {len(st)}")
