import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from airsim_env import AirSimDroneEnv

# Create environment
env = make_vec_env(AirSimDroneEnv, n_envs=4)

# Train RL model using PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

# Save model
model.save("models/ppo_drone")
print("Model saved successfully!")
