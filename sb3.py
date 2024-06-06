from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
from env import SinglePlayerTetris

def env_wrap():
    return SinglePlayerTetris(fast_soft=True, draw_ghost=True)

tmp_path = "./sb3_log/"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

tetris_env = make_vec_env(env_wrap, n_envs=8)
check_env(SinglePlayerTetris(fast_soft=True, draw_ghost=True))
model = PPO('MultiInputPolicy', tetris_env, verbose=1)
model.set_logger(new_logger)

checkpoint = CheckpointCallback(
    save_freq=1e6,
    save_path=tmp_path,
    name_prefix="tetris_sb3_ppo_mlp_",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

model.learn(total_timesteps=1e7, progress_bar=True, callback=checkpoint,)
model.save(f"tetris_sb3_ppo_nenv_1e7")