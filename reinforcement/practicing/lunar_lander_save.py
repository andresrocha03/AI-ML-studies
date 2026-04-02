import gymnasium as gym
from stable_baselines3 import PPO
import os

modelstr = "PPO"

models_dir = "models/" + modelstr
logdir  = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make("LunarLander-v2", render_mode="human")
env.reset()

model = PPO("MlpPolicy", env, verbose=1,tensorboard_log=logdir)

TIMESTEPS = 10000
for i in range(1,30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=modelstr)
    model.save(f"{models_dir}/{TIMESTEPS*i}")    

# vec_env = model.get_env()
# obs = vec_env.reset()

# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = vec_env.step(action)
#     vec_env.render("human")
   


env.close()