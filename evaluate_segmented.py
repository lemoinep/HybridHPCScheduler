from pathlib import Path
from stable_baselines3 import PPO

from apic.model import APICModel
from apic.env import APICEnv


def run_episode(env, agent):
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    return total_reward, steps, info


if __name__ == "__main__":
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_world = APICModel()
    model_world.build_demo()
    env = APICEnv(model_world, trace_file=str(output_dir / "eval_trace.jsonl"))

    agent = PPO.load(str(output_dir / "apic_ppo_segmented"))

    total_reward, steps, info = run_episode(env, agent)

    if hasattr(env, "export_segments_csv"):
        env.export_segments_csv(output_dir / "segments_eval.csv")
    if hasattr(env, "export_segments_gantt"):
        env.export_segments_gantt(output_dir / "gantt_eval.png")

    summary = {
        "total_reward": total_reward,
        "steps": steps,
        "final_info": info,
    }

    with open(output_dir / "evaluation_summary.txt", "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    env.close()
    print(summary)