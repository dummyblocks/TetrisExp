from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from env import SinglePlayerTetris

def env_wrap():
    return SinglePlayerTetris(fast_soft=True, draw_ghost=True)

tmp_path = "/sb3_ppo_log/"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

tetris_env = make_vec_env(env_wrap, n_envs=8)
check_env(SinglePlayerTetris(fast_soft=True, draw_ghost=True))
model = PPO('MultiInputPolicy', tetris_env, verbose=1)
model.set_logger(new_logger)
model.learn(total_timesteps=(1e+7))
model.save("tetris_sb3_ppo_2e7_nenv")