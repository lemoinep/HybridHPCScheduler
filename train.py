from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from apic.model import APICModel
from apic.env import APICEnv


def make_env():
    model = APICModel()
    model.build_demo()
    return APICEnv(model)


if __name__ == "__main__":
    env = make_env()
    check_env(env, warn=True)

    vec_env = DummyVecEnv([make_env])
    agent = PPO("MultiInputPolicy", vec_env, verbose=1)
    agent.learn(total_timesteps=5000)
    agent.save("apic_ppo")