from stable_baselines3 import PPO
from apic.model import APICModel
from apic.env import APICEnv


if __name__ == "__main__":
    world = APICModel()
    world.build_demo()
    env = APICEnv(world)

    agent = PPO.load("apic_ppo")
    obs, info = env.reset()

    total_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    print("Total reward:", total_reward)
    env.close()