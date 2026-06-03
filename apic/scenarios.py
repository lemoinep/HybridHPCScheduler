from dataclasses import dataclass
from typing import List, Literal, Dict, Any
ActionType = Literal["fail", "recover"]

@dataclass
class FailureEvent:
    time: int         
    device: str        
    action: ActionType 

class FailureScenario:
    def __init__(self, failures: List[FailureEvent]):
        self.failures: List[FailureEvent] = sorted(failures, key=lambda e: e.time)
        self.index: int = 0

    @classmethod
    def from_dicts(cls, events: List[Dict[str, Any]]) -> "FailureScenario":
        failures = [
            FailureEvent(
                time=int(e["time"]),
                device=str(e["device"]),
                action=e["action"],
            )
            for e in events
        ]
        return cls(failures)

    def reset(self):
        self.index = 0

    def apply(self, env) -> None:
        if not self.failures or self.index >= len(self.failures):
            return
        while self.index < len(self.failures) and self.failures[self.index].time <= env.now:
            event = self.failures[self.index]

            if event.action == "fail":
                env.fail_device(event.device)
                print(f"[FailureScenario] Device {event.device} FAILED at t={env.now}")
            elif event.action == "recover":
                env.recover_device(event.device)
                print(f"[FailureScenario] Device {event.device} RECOVERED at t={env.now}")

            self.index += 1