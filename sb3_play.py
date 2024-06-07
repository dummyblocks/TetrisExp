from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from env import SinglePlayerTetris

tetris_env = SinglePlayerTetris(render_mode='human',fast_soft=False, draw_ghost=True, auto_drop=False,fps=1)
model = PPO.load('sb3_log/tetris_sb3_ppo_image_10000000_steps.zip', tetris_env)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)

print(mean_reward, std_reward)

# Enjoy trained agent
# obs = tetris_env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, trunc, info = tetris_env.step(action)
#     tetris_env.render()