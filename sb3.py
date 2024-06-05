from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from env import SinglePlayerTetris
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
import tqdm
seed = 42
tmp_path = "./sb3_log/"
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

tetris_env = make_vec_env(SinglePlayerTetris, n_envs=8,seed = seed)
check_env(SinglePlayerTetris())
model = PPO('MultiInputPolicy', tetris_env, verbose=1,seed=seed)
model.set_logger(new_logger)

checkpoint_callback = CheckpointCallback(
  save_freq=2e5,
  save_path="./sb3_log/",
  name_prefix="tetris_sb3_ppo_image_",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

total_timesteps = 1e7
model.learn(total_timesteps=total_timesteps, progress_bar=True,  callback=checkpoint_callback)
model.save(f"tetris_sb3_ppo_image_{total_timesteps}")

