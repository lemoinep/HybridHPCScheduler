# To do: Integrate the loading of the hardware configuration model (CPU, GPU, TPU, NPU, DPU) 
# to maximize system optimization.
# Loading the exascale mapping, for example. To be defined.

from dataclasses import dataclass
from typing import Dict, List
import networkx as nx


@dataclass
class MemoryLevel:
    name: str
    capacity: int
    latency: int
    bandwidth: int


@dataclass
class Device:
    name: str
    kind: str
    speed: float
    threads: int
    memories: List[MemoryLevel]


@dataclass
class Task:
    name: str
    compute: int
    memory: int
    data_in: int
    data_out: int
    preferred_kind: str = ""
    priority: int = 1
    deadline: int = 10**9
    preemptible: bool = True
    segment_ms: int = 1000
    progress: int = 0
    current_device: str = ""


@dataclass
class Dep:
    src: str
    dst: str
    bytes_mb: int


@dataclass
class PlannedSegment:
    task: str
    segment_id: int
    device: str
    start: int
    end: int
    duration: int
    compute: int
    migrated: bool = False
    resumed: bool = False


class APICModel:
    def __init__(self):
        self.devices: Dict[str, Device] = {}
        self.tasks: Dict[str, Task] = {}
        self.deps: List[Dep] = []
        self.links = nx.MultiDiGraph()

    def add_device(self, name, kind, speed, threads, memories):
        self.devices[name] = Device(name, kind, speed, threads, memories)
        self.links.add_node(name, kind=kind)

    def add_link(self, src, dst, bandwidth, latency, kind="fabric"):
        self.links.add_edge(src, dst, bandwidth=bandwidth, latency=latency, kind=kind)

    def add_task(self, name, compute, memory, data_in, data_out,
                 preferred_kind="", priority=1, deadline=10**9, preemptible=True):
        self.tasks[name] = Task(
            name, compute, memory, data_in, data_out,
            preferred_kind, priority, deadline, preemptible
        )

    def add_dep(self, src, dst, bytes_mb):
        self.deps.append(Dep(src, dst, bytes_mb))

    def build_demo(self):
        # Todo: init parser
        self.add_device("CPU0", "cpu", 120, 64, [MemoryLevel("DRAM", 131072, 80, 200), MemoryLevel("L3", 32768, 12, 800)])
        self.add_device("CPU1", "cpu", 120, 64, [MemoryLevel("DRAM", 131072, 80, 200), MemoryLevel("L3", 32768, 12, 800)])
        self.add_device("GPU0", "gpu", 1200, 256, [MemoryLevel("HBM", 49152, 3, 1600), MemoryLevel("L2", 8192, 1, 3000)])
        self.add_device("GPU1", "gpu", 1100, 256, [MemoryLevel("HBM", 49152, 3, 1600), MemoryLevel("L2", 8192, 1, 3000)])
        self.add_device("NPU0", "npu", 800, 128, [MemoryLevel("SRAM", 16384, 2, 1400), MemoryLevel("CACHE", 4096, 1, 2500)])
        self.add_device("TPU0", "tpu", 950, 128, [MemoryLevel("HBM", 32768, 3, 1500), MemoryLevel("CACHE", 4096, 1, 2500)])
        self.add_device("DPU0", "dpu", 250, 32, [MemoryLevel("SRAM", 8192, 2, 1000), MemoryLevel("CACHE", 2048, 1, 1500)])

        for a in self.devices:
            for b in self.devices:
                if a == b:
                    continue
                ak = self.devices[a].kind
                bk = self.devices[b].kind
                bw = 2000 if {ak, bk} <= {"gpu", "tpu"} else 800 if "npu" in {ak, bk} else 250 if "dpu" in {ak, bk} else 120
                lat = 1 if bw >= 1500 else 3 if bw >= 800 else 8
                self.add_link(a, b, bw, lat, "xgmi" if "gpu" in {ak, bk} or "tpu" in {ak, bk} else "pci")

        self.add_task("ingest", 40, 512, 64, 64, "dpu", 3, 4000)
        self.add_task("decode", 80, 1024, 128, 128, "cpu", 3, 6000)
        self.add_task("preprocess", 120, 2048, 256, 256, "cpu", 4, 8000)
        self.add_task("infer_a", 500, 4096, 512, 256, "gpu", 10, 12000)
        self.add_task("infer_b", 420, 4096, 512, 256, "npu", 9, 12000)
        self.add_task("fuse", 90, 1024, 128, 128, "tpu", 8, 15000)
        self.add_task("postprocess", 70, 1024, 128, 128, "cpu", 6, 18000)
        self.add_task("store", 30, 512, 64, 64, "dpu", 2, 20000)

        self.add_dep("ingest", "decode", 64)
        self.add_dep("decode", "preprocess", 128)
        self.add_dep("preprocess", "infer_a", 512)
        self.add_dep("preprocess", "infer_b", 512)
        self.add_dep("infer_a", "fuse", 128)
        self.add_dep("infer_b", "fuse", 128)
        self.add_dep("fuse", "postprocess", 64)
        self.add_dep("postprocess", "store", 32)