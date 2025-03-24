from stable_baselines3 import PPO
from airsim_env import AirSimDroneEnv

# Load trained model
model = PPO.load("models/ppo_drone")

# Create environment
env = AirSimDroneEnv()
obs = env.reset()

# Test the agent
for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    if done:
        obs = env.reset()
