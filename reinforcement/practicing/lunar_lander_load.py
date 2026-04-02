import gymnasium as gym
from stable_baselines3 import PPO, A2C

modelstr = "A2C"

models_dir = "models/" + modelstr
model_path = f"{models_dir}/60000.zip"

env = gym.make("LunarLander-v2", render_mode="human")

model = PPO.load(model_path, env=env)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs,deterministic=True)
    observation, reward, done, info = vec_env.step(action)
    vec_env.render("human")
   


env.close()