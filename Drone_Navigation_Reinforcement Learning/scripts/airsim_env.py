import airsim
import gym
from gym import spaces
import numpy as np
import time


class AirSimDroneEnv(gym.Env):
    """Custom OpenAI Gym environment for AirSim Drone Navigation"""

    def __init__(self):
        super(AirSimDroneEnv, self).__init__()

        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Define action space (Up, Down, Left, Right, Forward, Backward)
        self.action_space = spaces.Discrete(6)

        # Define observation space (x, y, z, velocity_x, velocity_y, velocity_z)
        self.observation_space = spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32)

    def step(self, action):
        """Apply an action and return new state, reward, done flag"""
        if action == 0:
            self.client.moveByVelocityAsync(0, 0, -2, 1)  # Move Up
        elif action == 1:
            self.client.moveByVelocityAsync(0, 0, 2, 1)  # Move Down
        elif action == 2:
            self.client.moveByVelocityAsync(-2, 0, 0, 1)  # Left
        elif action == 3:
            self.client.moveByVelocityAsync(2, 0, 0, 1)  # Right
        elif action == 4:
            self.client.moveByVelocityAsync(0, 2, 0, 1)  # Forward
        elif action == 5:
            self.client.moveByVelocityAsync(0, -2, 0, 1)  # Backward

        time.sleep(1)  # Wait for movement to complete

        # Get new state
        state = self.get_drone_state()

        # Reward system
        reward = 10 if not self.client.simGetCollisionInfo().has_collided else -100
        done = self.client.simGetCollisionInfo().has_collided  # Episode ends on collision

        return state, reward, done, {}

    def reset(self):
        """Reset the environment"""
        self.client.reset()
        time.sleep(1)
        return self.get_drone_state()

    def get_drone_state(self):
        """Fetch drone telemetry data"""
        kinematics = self.client.getMultirotorState().kinematics_estimated
        position = kinematics.position
        velocity = kinematics.linear_velocity
        return np.array(
            [position.x_val, position.y_val, position.z_val, velocity.x_val, velocity.y_val, velocity.z_val])

    def render(self, mode="human"):
        pass
