# Author(s): Dr. Patrick Lemoine
# HybridHPCScheduler is an experimental framework for the development of an AI-based APIC system.
# It focuses on intelligent scheduling and resource orchestration in heterogeneous HPC environments.

import argparse
from pathlib import Path
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from apic.model import APICModel
from apic.env import APICEnv


def make_world():
    model = APICModel()
    model.build_demo()
    return model


def make_env(trace_file="output/trace.jsonl"):
    return APICEnv(make_world(), trace_file=trace_file)


def run_train(output_dir):
    env = make_env(trace_file=str(output_dir / "train_trace.jsonl"))
    check_env(env, warn=True)

    vec_env = DummyVecEnv([lambda: make_env(trace_file=str(output_dir / "train_vec_trace.jsonl"))])
    agent = PPO("MultiInputPolicy", vec_env, verbose=1)
    agent.learn(total_timesteps=5000)
    agent.save(str(output_dir / "apic_ppo_segmented"))
    env.close()


def run_evaluate(output_dir):
    agent = PPO.load(str(output_dir / "apic_ppo_segmented"))
    env = make_env(trace_file=str(output_dir / "eval_trace.jsonl"))

    obs, info = env.reset()
    total_reward = 0.0
    terminated = False
    truncated = False
    steps = 0

    while not (terminated or truncated):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    if hasattr(env, "export_segments_csv"):
        env.export_segments_csv(output_dir / "segments_eval.csv")
    if hasattr(env, "export_segments_gantt"):
        env.export_segments_gantt(output_dir / "gantt_eval.png")

    summary = {
        "total_reward": total_reward,
        "steps": steps,
        "time": info.get("time"),
        "active_tasks": info.get("active_tasks"),
        "completed_tasks": info.get("completed_tasks"),
        "late_tasks": info.get("late_tasks"),
        "congestion": info.get("congestion"),
        "replanned": info.get("replanned"),
        "migration_cost": info.get("migration_cost"),
        "continuity_bonus": info.get("continuity_bonus"),
        "segments": len(getattr(env, "segments_log", [])),
        "migrations": sum(1 for s in getattr(env, "segments_log", []) if s.get("migrated", False)),
    }

    pd.DataFrame([summary]).to_csv(output_dir / "evaluation_summary.csv", index=False)
    env.close()
    return summary


def run_benchmark(output_dir, n_episodes=10):
    agent = PPO.load(str(output_dir / "apic_ppo_segmented"))
    results = []

    for ep in range(n_episodes):
        env = make_env(trace_file=str(output_dir / f"benchmark_trace_ep{ep}.jsonl"))
        obs, info = env.reset()
        total_reward = 0.0
        terminated = False
        truncated = False
        steps = 0

        while not (terminated or truncated):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        segments = getattr(env, "segments_log", [])
        results.append({
            "episode": ep,
            "total_reward": total_reward,
            "steps": steps,
            "time": info.get("time"),
            "active_tasks": info.get("active_tasks"),
            "completed_tasks": info.get("completed_tasks"),
            "late_tasks": info.get("late_tasks"),
            "congestion": info.get("congestion"),
            "replanned": info.get("replanned"),
            "migration_cost": info.get("migration_cost"),
            "continuity_bonus": info.get("continuity_bonus"),
            "segments": len(segments),
            "migrations": sum(1 for s in segments if s.get("migrated", False)),
        })
        env.close()

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "benchmark_summary.csv", index=False)

    stats = []
    for col in ["total_reward", "steps", "late_tasks", "congestion", "replanned", "migration_cost", "continuity_bonus", "segments", "migrations"]:
        stats.append({
            "metric": col,
            "mean": float(df[col].mean()),
            "std": float(df[col].std(ddof=0)),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
        })

    pd.DataFrame(stats).to_csv(output_dir / "benchmark_stats.csv", index=False)
    return df, pd.DataFrame(stats)


def run_plot(output_dir):
    import matplotlib.pyplot as plt

    df = pd.read_csv(output_dir / "benchmark_summary.csv")

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    axes[0].plot(df["episode"], df["total_reward"], marker="o")
    axes[0].set_title("Reward par épisode")
    axes[0].set_xlabel("Épisode")
    axes[0].set_ylabel("Reward")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["episode"], df["late_tasks"], marker="o", color="red")
    axes[1].set_title("Tâches en retard")
    axes[1].set_xlabel("Épisode")
    axes[1].set_ylabel("Late tasks")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df["episode"], df["congestion"], marker="o", color="orange")
    axes[2].set_title("Congestion")
    axes[2].set_xlabel("Épisode")
    axes[2].set_ylabel("Congestion")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(df["episode"], df["migrations"], marker="o", color="purple")
    axes[3].set_title("Migrations")
    axes[3].set_xlabel("Épisode")
    axes[3].set_ylabel("Migrations")
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(df["episode"], df["segments"], marker="o", color="green")
    axes[4].set_title("Segments")
    axes[4].set_xlabel("Épisode")
    axes[4].set_ylabel("Segments")
    axes[4].grid(True, alpha=0.3)

    axes[5].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "benchmark_metrics.png", dpi=180)
    plt.close(fig)


def run_report(output_dir):
    df = pd.read_csv(output_dir / "benchmark_summary.csv")
    stats = pd.read_csv(output_dir / "benchmark_stats.csv")

    with open(output_dir / "benchmark_report.txt", "w", encoding="utf-8") as f:
        f.write("Benchmark report\n\n")
        f.write("Per-episode summary:\n")
        f.write(df.to_string(index=False))
        f.write("\n\nAggregate stats:\n")
        f.write(stats.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "evaluate", "benchmark", "plot", "report"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=str, default="output")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "train":
        run_train(output_dir)
    elif args.mode == "evaluate":
        summary = run_evaluate(output_dir)
        print(summary)
    elif args.mode == "benchmark":
        df, stats = run_benchmark(output_dir, n_episodes=args.episodes)
        print(df.to_string(index=False))
        print(stats.to_string(index=False))
    elif args.mode == "plot":
        run_plot(output_dir)
    elif args.mode == "report":
        run_report(output_dir)


if __name__ == "__main__":
    main()