import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .scheduler import APICScheduler
from .model import APICModel
import os
import time

DEBUG_LOG = "env_debug.log"


class APICEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, model: APICModel, tick_ms: int = 1000, episode_limit: int = 30000, trace_file="output/trace.jsonl"):
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
        self.segment_counter = 0
        self.episode_offset = 0

        self.device_names = list(self.model.devices.keys())
        self.task_names = list(self.model.tasks.keys())
        
        self._device_failure_pending = False

        self.action_space = spaces.MultiDiscrete([
            6,
            max(1, len(self.task_names)),
            max(1, len(self.device_names)),
            5,
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

    def _refresh_device_names(self):
        self.device_names = [d.name for d in self.model.get_online_devices()]

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

        active = self.schedule[(self.schedule.start <= self.now) & (self.schedule.end > self.now)]
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
    
    
    def fail_device(self, name: str):
        if name not in self.model.devices:
            print(f"[FAIL_DEVICE] Unknown device: {name}")
            return
    
        self.model.fail_device(name)
        self._refresh_device_names()
    
        affected = [
            seg for seg in self.segments_log
            if seg.get("device") == name and seg.get("end", self.now) > self.now
        ]
    
        # Forcer une replanification au prochain step
        self._device_failure_pending = True
        self.solve_failed = False
    
        self._log({
            "event": "device_failure",
            "time": int(self.now),
            "device": name,
            "affected_segments": len(affected),
        })

    def recover_device(self, name: str):
        if name not in self.model.devices:
            print(f"[RECOVER_DEVICE] Unknown device: {name}")
            return
    
        self.model.recover_device(name)
        self._refresh_device_names()
    
        self._log({
            "event": "device_recovery",
            "time": int(self.now),
            "device": name,
        })


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
    
        self.now = 0
        self.segments_log = []
        self.segment_states = {}
        self.solve_failed = False
        self.segment_counter = 0
        self._device_failure_pending = False
        
        self.episode_offset = int(np.random.randint(0, 3))
        self.random_priority_shift = int(np.random.randint(-1, 2))
        self.random_deadline_shift = int(np.random.randint(-500, 501))
    
        if len(self.scheduler.model.tasks) > 0:
            first_task_key = list(self.scheduler.model.tasks.keys())[0]
            first_task = self.scheduler.model.tasks[first_task_key]
            first_task.priority = max(1, min(10, first_task.priority + self.random_priority_shift))
            first_task.deadline = max(1000, first_task.deadline + self.random_deadline_shift)
    
        t0 = time.perf_counter()
        self.schedule = self.scheduler.solve_window(horizon=self.episode_limit)
        t1 = time.perf_counter()
        
        if self.schedule is not None:
            for idx, _ in self.schedule.iterrows():
                self.segment_states[idx] = 0
        else:
            print("WARNING: initial solve_window returned None in reset()")
    
        obs = self._observe(replanned=0)
        info = {"action_mask": None}
        t2 = time.perf_counter()
    
        #print(f"[RESET] solve={t1 - t0:.4f}s observe={t2 - t1:.4f}s total={t2 - t0:.4f}s")
    
        self._log({
            "event": "reset",
            "time": int(self.now),
            "obs": {k: int(v[0]) for k, v in obs.items()},
            "episode_offset": int(self.episode_offset),
            "priority_shift": int(self.random_priority_shift),
            "deadline_shift": int(self.random_deadline_shift),
        })
    
        return obs, info


    def apply_action(self, action):
        if self.schedule is None:
            return 0, 0.0, 0.0
    
        self._refresh_device_names()
        if not self.device_names:
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
                    self.segment_counter += 1
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
                    self.segment_counter += 1
                    replanned = 1
    
        elif action_type == 3:
            task.priority = min(task.priority + 1, 10)
            # Pas de replanification immédiate
    
        elif action_type == 4:
            task.priority = max(task.priority - 1, 1)
            # Pas de replanification immédiate
    
        elif action_type == 5:
            replanned = 1
    
        # Continuity bonus
        if action_type == 0:
            continuity_bonus = 0.0
        elif task.current_device and task.current_device == old_device_name:
            continuity_bonus = 1.0
    
        return replanned, migration_cost, continuity_bonus


    def step(self, action):
        t0 = time.perf_counter()
        
        replanned, migration_cost, continuity_bonus = self.apply_action(action)
        self.now += self.tick_ms
    
        t1 = time.perf_counter()
        
        if getattr(self, "_device_failure_pending", False):
           replanned = 1
           self._device_failure_pending = False
    
        if replanned == 1 and not getattr(self, "solve_failed", False):
            self.schedule = self.scheduler.solve_window(horizon=self.episode_limit)
            
            if self.schedule is not None:
                for idx, _ in self.schedule.iterrows():
                    if idx not in self.segment_states:
                        self.segment_states[idx] = 0
            else:
                self.solve_failed = True
    
        t2 = time.perf_counter()
    
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
            + 0.01 * float(continuity_bonus)
            + 0.15 * float(progress_bonus)
            + 0.3 * float(completion_bonus)
        )
    
        terminated = self.now >= int(self.schedule["end"].max())
        truncated = self.now >= self.episode_limit
    
        obs = self._observe(replanned=replanned)
        t3 = time.perf_counter()
    
        #if replanned:
        #    print(f"[STEP] apply={t1-t0:.4f}s solve={t2-t1:.4f}s compute={t3-t2:.4f}s total={t3-t0:.4f}s")
    
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
            "segment_counter": int(self.segment_counter),
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
            ax.barh(label, row["duration"], left=row["start"], color=colors[i], edgecolor="black")
            if bool(row["migrated"]):
                ax.text(row["start"] + row["duration"] / 2, i, "M", ha="center", va="center", color="white", fontsize=8)

        ax.set_xlabel("Time")
        ax.set_ylabel("Task segments")
        ax.set_title("Segment-aware APIC schedule")
        ax.grid(True, axis="x", linestyle="--", alpha=0.35)
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(filename, dpi=180)
        plt.close(fig)