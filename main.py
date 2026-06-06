# Author(s): Dr. Patrick Lemoine
# HybridHPCScheduler is an experimental framework for the development of an AI-based APIC system.
# It focuses on intelligent scheduling and resource orchestration in heterogeneous HPC environments.

# Todo : add FailureScenario

import argparse
from pathlib import Path
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from apic.model import APICModel, create_model
from apic.env import APICEnv
from apic.scenarios import FailureScenario


def make_world(config_path=None):
    return create_model(config_path)


#def make_env(trace_file="output/trace.jsonl", config_path=None):
#    return APICEnv(make_world(config_path), trace_file=trace_file)

def make_env(trace_file="output/trace.jsonl", config_path=None, failure_demo=False):
    env = APICEnv(make_world(config_path), trace_file=trace_file)
    if failure_demo:
        # Must be defined for FailureScenario
        env.failure_scenario = FailureScenario.from_dicts([
            {"time": 5000, "device": "GPU0", "action": "fail"},
            {"time": 15000, "device": "GPU0", "action": "recover"},
        ])
    return env

def run_show_config(config_path):
    from apic.model import print_config_summary
    print_config_summary(config_path)


def run_compare_configs(config_paths):
    from apic.model import compare_configs
    compare_configs(*config_paths)


def run_visualize(config_path, output_path):
    from apic.model import visualize_config
    visualize_config(config_path, output_path)


def run_visualize_all(config_dir, output_dir):
    from apic.model import visualize_all_configs
    visualize_all_configs(config_dir, output_dir)

def run_guide(config_path, output_path):
    from apic.model import generate_beginner_guide
    generate_beginner_guide(config_path, output_path)

#def run_train(output_dir, config_path=None):
#    env = make_env(trace_file=str(output_dir / "train_trace.jsonl"), config_path=config_path)
#    check_env(env, warn=True)

#    vec_env = DummyVecEnv([lambda: make_env(
#        trace_file=str(output_dir / "train_vec_trace.jsonl"),
#        config_path=config_path
#    )])
#    agent = PPO("MultiInputPolicy", vec_env, verbose=1)
#    agent.learn(total_timesteps=5000)
#    agent.save(str(output_dir / "apic_ppo_segmented"))
#    env.close()


def run_train(output_dir, config_path=None, failure_demo=False):
    env = make_env(
        trace_file=str(output_dir / "train_trace.jsonl"),
        config_path=config_path,
        failure_demo=failure_demo,
    )
    check_env(env, warn=True)

    vec_env = DummyVecEnv([lambda: make_env(
        trace_file=str(output_dir / "train_vec_trace.jsonl"),
        config_path=config_path
    )])
    agent = PPO("MultiInputPolicy", vec_env, verbose=1)
    agent.learn(total_timesteps=5000)
    agent.save(str(output_dir / "apic_ppo_segmented"))
    env.close()

def run_evaluate(output_dir, config_path=None):
    agent = PPO.load(str(output_dir / "apic_ppo_segmented"))
    env = make_env(trace_file=str(output_dir / "eval_trace.jsonl"), config_path=config_path)

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


def run_benchmark(output_dir, n_episodes=10, config_path=None):
    agent = PPO.load(str(output_dir / "apic_ppo_segmented"))
    results = []

    for ep in range(n_episodes):
        env = make_env(
            trace_file=str(output_dir / f"benchmark_trace_ep{ep}.jsonl"),
            config_path=config_path
        )
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


def run_benchmark_forced_migration(output_dir, n_episodes=5, config_path=None):
    results = []

    for ep in range(n_episodes):
        env = make_env(
            trace_file=str(output_dir / f"benchmark_trace_forced_ep{ep}.jsonl"),
            config_path=config_path
        )
        obs, info = env.reset()
        total_reward = 0.0
        terminated = False
        truncated = False
        steps = 0

        while not (terminated or truncated):
            # We start with a random action.
            action = env.action_space.sample()
            # Force the action type to "migrate to next device"
            action[0] = 1  # 1 = move to next device in your env

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
    df.to_csv(output_dir / "benchmark_forced_summary.csv", index=False)
    return df

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
    parser = argparse.ArgumentParser(
        description="APIC - AI Pipeline Intelligent Coordinator"
    )
    parser.add_argument(
        "mode",
        choices=["train", "evaluate", "benchmark", "plot", "report", 
                 "show", "compare", "visualize", "visualize-all", "guide"],
        help="Execution Mode"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes for benchmarking"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output Directory"
    )
    
    parser.add_argument(
       "--configs",
       nargs="+",
       default=None,
       help="List of configurations for comparison (compare mode)"
   )
    
    parser.add_argument(
       "--config_dir",
       type=str,
       default="configs",
       help="Folder containing the configurations (visualize-all mode)"
   )
    
    parser.add_argument(
        "--viz_output", 
        type=str,
        default=None,
        help="Output path for visualization"
   )
    
    parser.add_argument(
        "--failure_demo",
        action="store_true",
        help="Enable a demo device failure scenario during training/benchmark."
    )
    
    
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.config:
        print(f"Using configuration: {args.config}")
    else:
        print("Using default demo configuration (build_demo)")
        
        
    if args.mode == "visualize":
        if not args.config:
            print("❌ --config required for 'visualize' mode")
            return
        default_output = output_dir / f"{Path(args.config).stem}_visualization.png"
        output_path = getattr(args, 'viz_output', None) or str(default_output)
        run_visualize(args.config, output_path)
    
    elif args.mode == "visualize-all":
        default_viz_dir = output_dir / "visualizations"
        viz_output_dir = getattr(args, 'viz_output', None) or str(default_viz_dir)
        run_visualize_all(args.config_dir, viz_output_dir)
        
    elif args.mode == "guide":
        if not args.config:
            print("❌ --config required for 'guide' mode")
            return
        default_output = output_dir / f"{Path(args.config).stem}_guide.txt"
        guide_output = args.viz_output or str(default_output)
        run_guide(args.config, guide_output)

    elif args.mode == "show":
        if not args.config:
            print("❌ --config required for 'show' mode")
            return
        run_show_config(args.config)
    
    elif args.mode == "compare":
        if not args.configs or len(args.configs) < 2:
            print("❌ --configs <file1> <file2> ... required for 'compare' mode")
            return
        run_compare_configs(args.configs)    

    elif args.mode == "train":
        run_train(output_dir, config_path=args.config)
        
    elif args.mode == "evaluate":
        summary = run_evaluate(output_dir, config_path=args.config)
        print(summary)
        
    elif args.mode == "benchmark":
        df, stats = run_benchmark(output_dir, n_episodes=args.episodes, config_path=args.config)
        print(df.to_string(index=False))
        print(stats.to_string(index=False))
        
    elif args.mode == "plot":
        run_plot(output_dir)
        
    elif args.mode == "report":
        run_report(output_dir)
        
    #elif args.mode == "scan_exascale":
    #    run_report(output_dir)


if __name__ == "__main__":
    main()