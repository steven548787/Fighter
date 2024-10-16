from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
from env import CosmicHeatEnv

def train_model():
    env = DummyVecEnv([lambda: CosmicHeatEnv()])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    model.save("ppo_cosmic_heat")

if __name__ == "__main__":
    train_model()
