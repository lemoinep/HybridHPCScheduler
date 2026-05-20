from dataclasses import dataclass


@dataclass
class APICAction:
    action_type: int
    task_id: int = 0
    device_id: int = 0
    priority_delta: int = 2