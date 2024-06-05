from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from env import SinglePlayerTetris

tetris_env = make_vec_env(SinglePlayerTetris, n_envs=8)
check_env(SinglePlayerTetris())
model = PPO('MultiInputPolicy', tetris_env, verbose=1)
model.learn(total_timesteps=(2e+7))
model.save("tetris_sb3_ppo_2e7_n")