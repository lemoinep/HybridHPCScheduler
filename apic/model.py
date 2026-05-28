# To do: Integrate the loading of the hardware configuration model (CPU, GPU, TPU, NPU, DPU) 
# to maximize system optimization.
# Loading the exascale mapping, for example. To be defined.


from dataclasses import dataclass
from typing import Dict, List
import networkx as nx
import yaml
from pathlib import Path



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
        
    def load_from_config(self, config_path):
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Load devices
        if 'devices' in config:
            for dev in config['devices']:
                memories = []
                if 'memories' in dev:
                    for mem in dev['memories']:
                        memories.append(MemoryLevel(
                            name=mem['name'],
                            capacity=mem.get('capacity', 0),
                            latency=mem.get('latency', 0),
                            bandwidth=mem.get('bandwidth', 0)
                        ))
                
                self.add_device(
                    name=dev['name'],
                    kind=dev.get('kind', 'cpu'),
                    speed=dev.get('speed', 100),
                    threads=dev.get('threads', 1),
                    memories=memories
                )
        
        # Load topology
        if 'topology' in config:
            for link in config['topology']:
                if len(link) >= 3:
                    self.add_link(
                        src=link[0],
                        dst=link[1],
                        bandwidth=link[2],
                        latency=link[3] if len(link) > 3 else 5,
                        kind=link[4] if len(link) > 4 else "fabric"
                    )
        
        # Load Tasks
        if 'tasks' in config:
            for task in config['tasks']:
                self.add_task(
                    name=task['name'],
                    compute=task.get('compute', 100),
                    memory=task.get('memory', 1024),
                    data_in=task.get('data_in', 0),
                    data_out=task.get('data_out', 0),
                    preferred_kind=task.get('preferred_kind', ''),
                    priority=task.get('priority', 5),
                    deadline=task.get('deadline', 10000),
                    preemptible=task.get('preemptible', True)
                )
                
                # 
                if 'device' in task:
                    self.tasks[task['name']].current_device = task['device']
        
        # Load dependencies
        if 'dependencies' in config:
            for dep in config['dependencies']:
                self.add_dep(
                    src=dep['src'],
                    dst=dep['dst'],
                    bytes_mb=dep.get('bytes', 0)
                )
        
        return self
    
    
    
def load_model_from_config(config_path):
    model = APICModel()
    model.load_from_config(config_path)
    return model


def create_model(config_path=None):
    model = APICModel()
    
    if config_path:
        model.load_from_config(config_path)
    else:
        model.build_demo()
    
    return model


def print_config_summary(config_path):
    config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("=" * 70)
    print(f"📄 Configuration: {config_path.name}")
    print("=" * 70)
    
    # Devices
    if 'devices' in config:
        print(f"\n🖥️  DEVICES ({len(config['devices'])} total)")
        print("-" * 70)
        
        devices_by_kind = {}
        for dev in config['devices']:
            kind = dev.get('kind', 'unknown')
            if kind not in devices_by_kind:
                devices_by_kind[kind] = []
            devices_by_kind[kind].append(dev)
        
        for kind, devs in sorted(devices_by_kind.items()):
            print(f"\n  {kind.upper()} ({len(devs)})")
            for dev in devs:
                memories = dev.get('memories', [])
                total_mem = sum(m.get('capacity', 0) for m in memories)
                print(f"    • {dev['name']:<10} speed={dev.get('speed', 0):>6}  "
                      f"threads={dev.get('threads', 0):>4}  "
                      f"memory={total_mem:>8} MB")
    
    # Topology
    if 'topology' in config:
        print(f"\n🔗 TOPOLOGY ({len(config['topology'])} links)")
        print("-" * 70)
        
        links_by_kind = {}
        for link in config['topology']:
            kind = link[4] if len(link) > 4 else 'unknown'
            if kind not in links_by_kind:
                links_by_kind[kind] = []
            links_by_kind[kind].append(link)
        
        for kind, links in sorted(links_by_kind.items()):
            print(f"\n  {kind.upper()} ({len(links)} links)")
            # Afficher quelques exemples
            for link in links[:5]:
                bw = link[2] if len(link) > 2 else 0
                lat = link[3] if len(link) > 3 else 0
                print(f"    • {link[0]:<10} ↔ {link[1]:<10}  "
                      f"BW={bw:>5} GB/s  latency={lat:>3} ns")
            if len(links) > 5:
                print(f"    ... and {len(links) - 5} more")
    
    # Tasks
    if 'tasks' in config:
        print(f"\n📋 TASKS ({len(config['tasks'])} total)")
        print("-" * 70)
        
        tasks_by_kind = {}
        for task in config['tasks']:
            kind = task.get('preferred_kind', 'any')
            if kind not in tasks_by_kind:
                tasks_by_kind[kind] = []
            tasks_by_kind[kind].append(task)
        
        for kind, tasks in sorted(tasks_by_kind.items()):
            print(f"\n  Preferred: {kind.upper()} ({len(tasks)} tasks)")
            for task in tasks:
                print(f"    • {task['name']:<15} "
                      f"compute={task.get('compute', 0):>5}  "
                      f"memory={task.get('memory', 0):>6} MB  "
                      f"priority={task.get('priority', 0):>2}  "
                      f"deadline={task.get('deadline', 0):>6} ms")
    
    # Dependencies
    if 'dependencies' in config:
        print(f"\n🔀 DEPENDENCIES ({len(config['dependencies'])} total)")
        print("-" * 70)
        for dep in config['dependencies']:
            print(f"    {dep['src']:<15} → {dep['dst']:<15}  "
                  f"{dep.get('bytes', 0):>5} MB")
    
    # Summary stats
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    
    num_devices = len(config.get('devices', []))
    num_tasks = len(config.get('tasks', []))
    num_links = len(config.get('topology', []))
    num_deps = len(config.get('dependencies', []))
    
    total_compute = sum(t.get('compute', 0) for t in config.get('tasks', []))
    total_memory = sum(t.get('memory', 0) for t in config.get('tasks', []))
    
    avg_deadline = (sum(t.get('deadline', 0) for t in config.get('tasks', [])) / num_tasks 
                    if num_tasks > 0 else 0)
    
    print(f"  Devices:       {num_devices:>4}")
    print(f"  Tasks:         {num_tasks:>4}")
    print(f"  Links:         {num_links:>4}")
    print(f"  Dependencies:  {num_deps:>4}")
    print(f"\n  Total compute: {total_compute:>6}")
    print(f"  Total memory:  {total_memory:>6} MB")
    print(f"  Avg deadline:  {avg_deadline:>6.0f} ms")
    
    print("=" * 70)
    print()


def compare_configs(*config_paths):
    configs = []
    names = []
    
    for path in config_paths:
        path = Path(path)
        if not path.exists():
            print(f"❌ Configuration file not found: {path}")
            continue
        
        with open(path, 'r', encoding='utf-8') as f:
            configs.append(yaml.safe_load(f))
            names.append(path.stem)
    
    if not configs:
        print("No valid configurations to compare")
        return
    
    print("=" * 100)
    print("📊 CONFIGURATION COMPARISON")
    print("=" * 100)
    
    # Header
    header = f"{'Metric':<25}"
    for name in names:
        header += f"{name:>20}"
    print(header)
    print("-" * 100)
    
    # Devices
    row = f"{'Devices':<25}"
    for cfg in configs:
        row += f"{len(cfg.get('devices', [])):>20}"
    print(row)
    
    # Tasks
    row = f"{'Tasks':<25}"
    for cfg in configs:
        row += f"{len(cfg.get('tasks', [])):>20}"
    print(row)
    
    # Links
    row = f"{'Topology links':<25}"
    for cfg in configs:
        row += f"{len(cfg.get('topology', [])):>20}"
    print(row)
    
    # Dependencies
    row = f"{'Dependencies':<25}"
    for cfg in configs:
        row += f"{len(cfg.get('dependencies', [])):>20}"
    print(row)
    
    print("-" * 100)
    
    # Total compute
    row = f"{'Total compute':<25}"
    for cfg in configs:
        total = sum(t.get('compute', 0) for t in cfg.get('tasks', []))
        row += f"{total:>20}"
    print(row)
    
    # Total memory
    row = f"{'Total memory (MB)':<25}"
    for cfg in configs:
        total = sum(t.get('memory', 0) for t in cfg.get('tasks', []))
        row += f"{total:>20}"
    print(row)
    
    # Avg deadline
    row = f"{'Avg deadline (ms)':<25}"
    for cfg in configs:
        tasks = cfg.get('tasks', [])
        avg = sum(t.get('deadline', 0) for t in tasks) / len(tasks) if tasks else 0
        row += f"{avg:>20.0f}"
    print(row)
    
    # Device types
    print("-" * 100)
    print("\nDevice breakdown:")
    
    all_kinds = set()
    for cfg in configs:
        for dev in cfg.get('devices', []):
            all_kinds.add(dev.get('kind', 'unknown'))
    
    for kind in sorted(all_kinds):
        row = f"  {kind.upper():<23}"
        for cfg in configs:
            count = sum(1 for d in cfg.get('devices', []) if d.get('kind') == kind)
            row += f"{count:>20}"
        print(row)
    
    print("=" * 100)
    print()
    
