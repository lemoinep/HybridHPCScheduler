import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .scheduler import APICScheduler
from .model import APICModel
import os

DEBUG_LOG = "env_debug.log"


class APICEnv(gym.Env):
    metadata = {"render_modes": []}
    

    def __init__(self, model: APICModel, tick_ms: int = 1000,
                 episode_limit: int = 30000, trace_file="output/trace.jsonl"):
        super().__init__()
        self.model = model
        self.scheduler = APICScheduler(model)
        self.tick_ms = tick_ms
        self.episode_limit = episode_limit
        self.trace_file = trace_file

        self.now = 0
        self.schedule = None
        self.segments_log = []
        self.segment_states = {}
        self.solve_failed = False

        self.device_names = list(self.model.devices.keys())
        self.task_names = list(self.model.tasks.keys())

        self.action_space = spaces.MultiDiscrete([
            6,                               # action_type (0..5)
            max(1, len(self.task_names)),    # task_id
            max(1, len(self.device_names)),  # device_id
            5,                               # priority_delta (0..4)
        ])

        self.observation_space = spaces.Dict({
            "time": spaces.Box(low=0, high=10**9, shape=(1,), dtype=np.int64),
            "active_tasks": spaces.Box(low=0, high=10**6, shape=(1,), dtype=np.int64),
            "completed_tasks": spaces.Box(low=0, high=10**6, shape=(1,), dtype=np.int64),
            "late_tasks": spaces.Box(low=0, high=10**6, shape=(1,), dtype=np.int64),
            "congestion": spaces.Box(low=0, high=10**6, shape=(1,), dtype=np.int64),
            "replanned": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int64),
        })

        os.makedirs(os.path.dirname(self.trace_file), exist_ok=True)
        self._trace = open(self.trace_file, "w", encoding="utf-8")

    
    def close(self):
        if getattr(self, "_trace", None):
            self._trace.close()

    def _log(self, rec):
        self._trace.write(json.dumps(rec) + "\n")
        self._trace.flush()

    def _record_segment(self, seg, action_type):
        self.segments_log.append({
            "task": seg.task,
            "segment_id": int(seg.segment_id),
            "device": seg.device,
            "start": int(seg.start),
            "end": int(seg.end),
            "duration": int(seg.duration),
            "compute": int(seg.compute),
            "migrated": bool(seg.migrated),
            "resumed": bool(seg.resumed),
            "action_type": int(action_type),
        })

    def _observe(self, replanned=0):
        if self.schedule is None:
            return {
                "time": np.array([self.now], dtype=np.int64),
                "active_tasks": np.array([0], dtype=np.int64),
                "completed_tasks": np.array([0], dtype=np.int64),
                "late_tasks": np.array([0], dtype=np.int64),
                "congestion": np.array([0], dtype=np.int64),
                "replanned": np.array([replanned], dtype=np.int64),
            }

        active = self.schedule[(self.schedule.start <= self.now) &
                               (self.schedule.end > self.now)]
        completed = self.schedule[self.schedule.end <= self.now]
        late = self.schedule[self.schedule.end > self.schedule.deadline]
        congestion = int((active.groupby("device").size() ** 2).sum()) if len(active) else 0

        return {
            "time": np.array([self.now], dtype=np.int64),
            "active_tasks": np.array([len(active)], dtype=np.int64),
            "completed_tasks": np.array([len(completed)], dtype=np.int64),
            "late_tasks": np.array([len(late)], dtype=np.int64),
            "congestion": np.array([congestion], dtype=np.int64),
            "replanned": np.array([replanned], dtype=np.int64),
        }


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.now = 0
        self.segments_log = []
        self.segment_states = {}
        self.solve_failed = False

        # Planning initial
        self.schedule = self.scheduler.solve_window(horizon=self.episode_limit)
        if self.schedule is None:
            print("ERROR: initial solve_window returned None in reset()")
            #raise RuntimeError("initial schedule is None in reset()")
        else :
            for idx, _ in self.schedule.iterrows():
                self.segment_states[idx] = 0

        obs = self._observe(replanned=0)
        info = {"action_mask": None}
        self._log({
            "event": "step",
            "time": int(self.now),
            "obs": {k: int(v[0]) for k, v in obs.items()},
        })
        return obs, info


    def apply_action(self, action):
        if self.schedule is None:
            return 0, 0.0, 0.0

        action = np.asarray(action).astype(int).tolist()
        action_type = int(action[0])

        replanned = 0
        migration_cost = 0.0
        continuity_bonus = 0.0

        task_name = self.task_names[0]
        task = self.model.tasks[task_name]

        if task.current_device in self.device_names:
            current_idx = self.device_names.index(task.current_device)
        else:
            current_idx = 0
            task.current_device = self.device_names[current_idx]

        old_device_name = task.current_device

        if action_type == 0:
            pass

        elif action_type == 1:
            device_id = min(current_idx + 1, len(self.device_names) - 1)
            device_name = self.device_names[device_id]
            if task.current_device != device_name:
                migration_cost = self.scheduler.migration_cost(task.current_device, device_name, task)
            task.current_device = device_name
            seg = self.scheduler.make_segment(task_name, self.now, device=device_name, migrated=True)
            if seg is not None:
                self.scheduler.apply_segment(seg)
                self._record_segment(seg, action_type)
                replanned = 1

        elif action_type == 2:
            device_id = max(current_idx - 1, 0)
            device_name = self.device_names[device_id]
            if task.current_device != device_name:
                migration_cost = self.scheduler.migration_cost(task.current_device, device_name, task)
            task.current_device = device_name
            seg = self.scheduler.make_segment(task_name, self.now, device=device_name, migrated=True)
            if seg is not None:
                self.scheduler.apply_segment(seg)
                self._record_segment(seg, action_type)
                replanned = 1

        elif action_type == 3:
            task.priority = min(task.priority + 1, 10)
            replanned = 1

        elif action_type == 4:
            task.priority = max(task.priority - 1, 1)
            replanned = 1

        elif action_type == 5:
            replanned = 1

        if replanned and self.schedule is not None:
            hints = {}
            for _, row in self.schedule.iterrows():
                hints[("x", row.task, row.device)] = 1
                hints[("s", row.task)] = int(row.start)

            new_schedule = self.scheduler.solve_window(
                horizon=self.episode_limit,
                hints=hints,
            )

            if new_schedule is not None:
                self.schedule = new_schedule

        if action_type == 0:
            continuity_bonus = 0.0
        elif task.current_device and task.current_device == old_device_name:
            continuity_bonus = 1.0

        return replanned, migration_cost, continuity_bonus

  
    def step(self, action):
        replanned, migration_cost, continuity_bonus = self.apply_action(action)
        self.now += self.tick_ms

        if getattr(self, "solve_failed", False):
            obs = self._observe(replanned=replanned)
            info = {
                "error": "solve_failed",
                "segments": len(self.segments_log),
                "action_mask": None,
            }
            return obs, -100.0, True, False, info

        if self.schedule is None:
            print("WARNING: schedule is None in step()")
            obs = self._observe(replanned=replanned)
            info = {
                "error": "no_schedule",
                "segments": len(self.segments_log),
                "action_mask": None,
            }
            return obs, -100.0, True, False, info

        active = self.schedule[(self.schedule.start <= self.now) & (self.schedule.end > self.now)]
        completed = self.schedule[self.schedule.end <= self.now]
        late = self.schedule[self.schedule.end > self.schedule.deadline]
        congestion = int((active.groupby("device").size() ** 2).sum()) if len(active) else 0

        progress_bonus = 0.0
        completion_bonus = 0.0

        for idx, seg in self.schedule.iterrows():
            prev_state = self.segment_states.get(idx, 0)
            current_state = 0

            if self.now >= seg["start"]:
                current_state = 1
            if self.now >= seg["end"]:
                current_state = 2

            if current_state > prev_state:
                progress_bonus += 1.0
            if current_state == 2 and prev_state < 2:
                completion_bonus += 1.0

            self.segment_states[idx] = current_state

        late_penalty = min(len(late), 10)
        cong_penalty = min(congestion, 20)
        mig_penalty = min(float(migration_cost), 100.0)

        reward = (
            -0.02 * float(cong_penalty)
            - 0.1 * float(late_penalty)
            - 0.01 * float(len(active))
            - 0.01 * float(mig_penalty)
            + 0.2 * float(replanned)
            + 0.02 * float(continuity_bonus)
            + 0.15 * float(progress_bonus)
            + 0.3 * float(completion_bonus)
        )

        terminated = self.now >= int(self.schedule["end"].max())
        truncated = self.now >= self.episode_limit

        obs = self._observe(replanned=replanned)

        info = {
            "time": int(self.now),
            "active_tasks": int(len(active)),
            "completed_tasks": int(len(completed)),
            "late_tasks": int(len(late)),
            "congestion": int(congestion),
            "replanned": int(replanned),
            "migration_cost": float(migration_cost),
            "continuity_bonus": float(continuity_bonus),
            "progress_bonus": float(progress_bonus),
            "completion_bonus": float(completion_bonus),
            "segments": int(len(self.segments_log)),
            "action_mask": None,
        }

        action_list = np.asarray(action).astype(int).tolist()
        self._log({
            "event": "step",
            "time": int(self.now),
            "action": action_list,
            "reward": float(reward),
            "info": info,
            "segments_last": self.segments_log[-1] if self.segments_log else None,
        })

        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write("----- STEP DEBUG -----\n")
            f.write(f"action = {action_list}\n")
            f.write(f"time = {self.now}\n")
            f.write(f"replanned = {replanned}\n")
            f.write(f"migration_cost = {migration_cost}\n")
            f.write(f"continuity_bonus = {continuity_bonus}\n")
            f.write(f"progress_bonus = {progress_bonus}\n")
            f.write(f"completion_bonus = {completion_bonus}\n")
            f.write(f"active = {len(active)}\n")
            f.write(f"late = {len(late)}\n")
            f.write(f"congestion = {congestion}\n")
            f.write(f"reward = {reward}\n")
            f.write("----------------------\n")

        return obs, reward, terminated, truncated, info
        
        
    def export_segments_csv(self, filename="output/segments.csv"):
        import pandas as pd
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        pd.DataFrame(self.segments_log).to_csv(filename, index=False)

    def export_segments_gantt(self, filename="output/gantt_segments.png"):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if not self.segments_log:
            return

        df = pd.DataFrame(self.segments_log)
        fig, ax = plt.subplots(figsize=(14, 7))
        df = df.sort_values(["start", "task", "segment_id"]).reset_index(drop=True)
        colors = plt.cm.tab20(np.linspace(0, 1, len(df)))

        for i, row in df.iterrows():
            label = f"{row['task']}:{row['segment_id']}"
            ax.barh(label, row["duration"], left=row["start"],
                    color=colors[i], edgecolor="black")
            if bool(row["migrated"]):
                ax.text(row["start"] + row["duration"] / 2, i, "M",
                        ha="center", va="center", color="white", fontsize=8)

        ax.set_xlabel("Time")
        ax.set_ylabel("Task segments")
        ax.set_title("Segment-aware APIC schedule")
        ax.grid(True, axis="x", linestyle="--", alpha=0.35)
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(filename, dpi=180)
        plt.close(fig)