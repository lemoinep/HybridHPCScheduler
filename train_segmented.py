from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from apic.model import APICModel
from apic.env import APICEnv


class APICTrainingWrapper:
    def __init__(self, env, log_dir="output"):
        self.env = env
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.episode_id = 0
        self.episode_reward = 0.0
        self.current_mask = None

    def reset(self, **kwargs):
        self.episode_reward = 0.0
        obs, info = self.env.reset(**kwargs)
        self.current_mask = info.get("action_mask", None)
        self._log_event("reset", {"obs": self._compact_obs(obs), "info": self._compact_info(info)})
        return obs, info

    def step(self, action):
        action = self._apply_mask(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += reward
        self.current_mask = info.get("action_mask", self.current_mask)
        self._log_event("step", {
            "action": self._compact_action(action),
            "reward": float(reward),
            "info": self._compact_info(info),
        })
        if terminated or truncated:
            self._finalize_episode(info)
        return obs, reward, terminated, truncated, info

    def _apply_mask(self, action):
        if self.current_mask is None:
            return action
        if isinstance(action, dict):
            a = dict(action)
            at = int(a.get("action_type", 0))
            mask = self.current_mask.get("action_type", None)
            if mask is not None and (at < 0 or at >= len(mask) or not mask[at]):
                a["action_type"] = 0
            return a
        return action

    def _finalize_episode(self, info):
        if hasattr(self.env, "export_segments_csv"):
            self.env.export_segments_csv(self.log_dir / f"segments_ep{self.episode_id}.csv")
        if hasattr(self.env, "export_segments_gantt"):
            self.env.export_segments_gantt(self.log_dir / f"gantt_ep{self.episode_id}.png")
        summary = {
            "episode_id": self.episode_id,
            "episode_reward": float(self.episode_reward),
            "final_info": self._compact_info(info),
        }
        self._log_event("episode_end", summary)
        self.episode_id += 1

    def _log_event(self, event_type, payload):
        import json
        record = {"event": event_type, **payload}
        with open(self.log_dir / "episodes.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _compact_obs(self, obs):
        if isinstance(obs, dict):
            out = {}
            for k, v in obs.items():
                if hasattr(v, "__len__") and len(v) == 1:
                    out[k] = int(v[0])
                else:
                    out[k] = str(v)
            return out
        return str(obs)

    def _compact_action(self, action):
        if isinstance(action, dict):
            out = {}
            for k, v in action.items():
                try:
                    out[k] = int(v)
                except Exception:
                    out[k] = v
            return out
        return str(action)

    def _compact_info(self, info):
        if isinstance(info, dict):
            out = {}
            for k, v in info.items():
                if hasattr(v, "tolist"):
                    out[k] = v.tolist()
                elif isinstance(v, (int, float, str, bool)) or v is None:
                    out[k] = v
                else:
                    out[k] = str(v)
            return out
        return str(info)

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()


def make_env():
    model = APICModel()
    model.build_demo()
    return APICEnv(model)


if __name__ == "__main__":
    env = make_env()
    check_env(env, warn=True)

    wrapped_env = APICTrainingWrapper(env)

    def wrapped_make_env():
        model = APICModel()
        model.build_demo()
        return APICTrainingWrapper(APICEnv(model))

    vec_env = DummyVecEnv([wrapped_make_env])

    agent = PPO("MultiInputPolicy", vec_env, verbose=1)
    agent.learn(total_timesteps=5000)
    agent.save("output/apic_ppo_segmented")

    wrapped_env.close()