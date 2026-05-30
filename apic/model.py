# To do: Integrate the loading of the hardware configuration model (CPU, GPU, TPU, NPU, DPU) 
# to maximize system optimization.
# Loading the exascale mapping, for example. To be defined.


from dataclasses import dataclass
from typing import Dict, List
import networkx as nx
import yaml
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import networkx as nx
import numpy as np


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
    



def visualize_config(config_path, output_path="config_visualization.png", dpi=300):
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    ax_devices = fig.add_subplot(gs[0, 0])
    ax_topology = fig.add_subplot(gs[0, 1])
    ax_tasks = fig.add_subplot(gs[1, :])
    
    fig.suptitle(f"HPC Configuration: {config_path.name}", 
                 fontsize=20, fontweight='bold', y=0.98)
    
    _plot_devices_memory(ax_devices, config)
    
    _plot_network_topology(ax_topology, config)
    
    _plot_task_dependencies(ax_tasks, config)
    
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved to: {output_path}")


def _plot_devices_memory(ax, config):
    ax.set_title("Devices & Memory Hierarchy", fontsize=14, fontweight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    devices = config.get('devices', [])
    if not devices:
        ax.text(5, 5, "No devices configured", ha='center', va='center')
        return
    
    # Colors by device type
    device_colors = {
        'cpu': '#4A90E2',
        'gpu': '#50C878',
        'npu': '#F39C12',
        'tpu': '#9B59B6',
        'dpu': '#E74C3C',
    }
    
    n_devices = len(devices)
    cols = min(4, n_devices)
    rows = (n_devices + cols - 1) // cols
    
    x_spacing = 9 / cols
    y_spacing = 9 / rows
    
    for idx, dev in enumerate(devices):
        row = idx // cols
        col = idx % cols
        
        x = 0.5 + col * x_spacing
        y = 9.5 - row * y_spacing
        
        kind = dev.get('kind', 'cpu')
        color = device_colors.get(kind, '#95A5A6')
        
        # Device's main box
        box = FancyBboxPatch(
            (x, y - 0.8), 2.0, 0.8,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor='black',
            linewidth=2,
            alpha=0.7
        )
        ax.add_patch(box)
        
        # Device Name
        ax.text(x + 1.0, y - 0.4, dev['name'], 
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        # Device Speed
        speed = dev.get('speed', 0)
        threads = dev.get('threads', 0)
        ax.text(x + 1.0, y - 0.6, f"{speed} TF | {threads} cores",
                ha='center', va='center', fontsize=7, color='white')
        
        # Memory Hierarchy
        memories = dev.get('memories', [])
        mem_y = y - 1.0
        
        for mem_idx, mem in enumerate(memories):
            mem_y -= 0.3
            mem_name = mem.get('name', 'MEM')
            mem_cap = mem.get('capacity', 0)
            mem_bw = mem.get('bandwidth', 0)
            
            mem_box = FancyBboxPatch(
                (x + 0.2, mem_y), 1.6, 0.25,
                boxstyle="round,pad=0.02",
                facecolor='white',
                edgecolor=color,
                linewidth=1.5,
                alpha=0.8
            )
            ax.add_patch(mem_box)
            
            ax.text(x + 1.0, mem_y + 0.125, 
                    f"{mem_name}: {mem_cap//1024}GB @ {mem_bw}GB/s",
                    ha='center', va='center', fontsize=6, color='black')


def _plot_network_topology(ax, config):
    ax.set_title("Network Topology", fontsize=14, fontweight='bold')
    
    devices = config.get('devices', [])
    topology = config.get('topology', [])
    
    if not devices:
        ax.text(0.5, 0.5, "No topology configured", ha='center', va='center', 
                transform=ax.transAxes)
        ax.axis('off')
        return
    
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Add the devices as nodes
    device_kinds = {}
    for dev in devices:
        G.add_node(dev['name'])
        device_kinds[dev['name']] = dev.get('kind', 'cpu')
    

    link_data = {}
    for link in topology:
        src, dst = link[0], link[1]
        bw = link[2] if len(link) > 2 else 100
        lat = link[3] if len(link) > 3 else 5
        kind = link[4] if len(link) > 4 else 'fabric'
        
        edge = tuple(sorted([src, dst]))
        if edge not in link_data:
            G.add_edge(src, dst, bandwidth=bw, latency=lat, kind=kind)
            link_data[edge] = (bw, lat, kind)
    

    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    device_colors = {
        'cpu': '#4A90E2',
        'gpu': '#50C878',
        'npu': '#F39C12',
        'tpu': '#9B59B6',
        'dpu': '#E74C3C',
    }
    
    node_colors = [device_colors.get(device_kinds.get(node, 'cpu'), '#95A5A6') 
                   for node in G.nodes()]
    
    # Draw the edges with a thickness proportional to the bandwidth
    max_bw = max((data['bandwidth'] for _, _, data in G.edges(data=True)), default=1)
    
    for (u, v, data) in G.edges(data=True):
        bw = data['bandwidth']
        lat = data['latency']
        kind = data.get('kind', 'fabric')
        
        # Thickness proportional to bandwidth
        width = 0.5 + 5 * (bw / max_bw)
        
        # Color according to the type of link
        link_colors = {
            'xgmi': '#E74C3C',
            'pci': '#3498DB',
            'fabric': '#95A5A6',
        }
        color = link_colors.get(kind, '#95A5A6')
        
        # Style according to latency (dotted line if high latency)
        style = 'dashed' if lat > 5 else 'solid'
        
        nx.draw_networkx_edges(G, pos, [(u, v)], width=width, 
                               edge_color=color, style=style, alpha=0.6, ax=ax)
    
    # Draw the knots
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                           node_size=1500, alpha=0.9, ax=ax)
    
    # Dessiner les labels
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', 
                            font_color='white', ax=ax)
    
    # Legend for link types
    legend_elements = [
        mpatches.Patch(facecolor='#E74C3C', label='XGMI (high-speed)'),
        mpatches.Patch(facecolor='#3498DB', label='PCIe'),
        mpatches.Patch(facecolor='#95A5A6', label='Fabric'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    ax.axis('off')
    
    
def _plot_task_dependencies(ax, config):
    """Visualise le DAG des tâches avec dépendances."""
    ax.set_title("Task Dependency Graph (DAG)", fontsize=14, fontweight='bold')
    
    tasks = config.get('tasks', [])
    dependencies = config.get('dependencies', [])
    
    if not tasks:
        ax.text(0.5, 0.5, "No tasks configured", ha='center', va='center',
                transform=ax.transAxes)
        ax.axis('off')
        return
    
    G = nx.DiGraph()
    
    # Add tasks as nodes with attributes
    task_info = {}
    for task in tasks:
        name = task['name']
        G.add_node(name)
        task_info[name] = {
            'compute': task.get('compute', 0),
            'memory': task.get('memory', 0),
            'priority': task.get('priority', 5),
            'deadline': task.get('deadline', 0),
            'kind': task.get('preferred_kind', 'cpu'),
        }
    
    # Add dependencies
    dep_weights = {}
    for dep in dependencies:
        src = dep['src']
        dst = dep['dst']
        bytes_val = dep.get('bytes', 0)
        G.add_edge(src, dst, weight=bytes_val)
        dep_weights[(src, dst)] = bytes_val
    
    # Hierarchical Layout
    try:
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    except:
        pos = nx.spring_layout(G, seed=42)
    
    # Node colors according to priority
    priorities = [task_info[node]['priority'] for node in G.nodes()]
    
    # Draw edges with weight labels
    max_weight = max(dep_weights.values(), default=1)
    
    for (u, v) in G.edges():
        weight = dep_weights.get((u, v), 0)
        width = 0.5 + 3 * (weight / max_weight)
        
        nx.draw_networkx_edges(G, pos, [(u, v)], width=width,
                               edge_color='#34495E', alpha=0.6,
                               arrowsize=20, arrowstyle='->', ax=ax)
        
        # Weight label on the edge
        edge_x = (pos[u][0] + pos[v][0]) / 2
        edge_y = (pos[u][1] + pos[v][1]) / 2
        ax.text(edge_x, edge_y, f"{weight}MB", fontsize=7,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Draw the nodes with color according to priority
    nodes = nx.draw_networkx_nodes(G, pos, node_color=priorities,
                                    cmap=plt.cm.RdYlGn_r, vmin=0, vmax=10,
                                    node_size=2000, alpha=0.9, ax=ax)
    
    if nodes is not None:
        # Add an outline to high-priority tasks
        high_priority_nodes = [node for node in G.nodes() 
                               if task_info[node]['priority'] >= 8]
        if high_priority_nodes:
            high_priority_pos = {k: v for k, v in pos.items() if k in high_priority_nodes}
            nx.draw_networkx_nodes(G, high_priority_pos, 
                                   nodelist=high_priority_nodes,
                                   node_color='none',
                                   edgecolors='red',
                                   linewidths=3,
                                   node_size=2200,
                                   ax=ax)
    
    
    # Node labels with info
    labels = {}
    for node in G.nodes():
        info = task_info[node]
        labels[node] = f"{node}\n{info['compute']}c | {info['memory']//1024}GB\nP{info['priority']}"
    
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight='bold', ax=ax)
    
    # Colorbar for priority (utilise 'nodes' pour garantir la cohérence)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, 
                               norm=plt.Normalize(vmin=0, vmax=10))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, aspect=30)
    cbar.set_label('Task Priority', fontsize=10)
    
    device_colors = {
        'cpu': '#4A90E2',
        'gpu': '#50C878',
        'npu': '#F39C12',
        'tpu': '#9B59B6',
        'dpu': '#E74C3C',
    }
    
    # Find the types of devices present
    kinds_present = set(task_info[node]['kind'] for node in G.nodes())
    
    legend_elements = []
    for kind in sorted(kinds_present):
        if kind in device_colors:
            legend_elements.append(
                mpatches.Patch(facecolor=device_colors[kind], 
                              label=f'{kind.upper()} tasks',
                              alpha=0.7)
            )
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8, 
                 title='Device')
    
    # Add DAG statistics in a corner
    num_tasks = len(G.nodes())
    num_deps = len(G.edges())
    total_data = sum(dep_weights.values())
    
    stats_text = f"Tasks: {num_tasks}\nDeps: {num_deps}\nData: {total_data}MB"
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    ax.axis('off')



def visualize_all_configs(config_dir="configs", output_dir="visualizations"):
    config_dir = Path(config_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    yaml_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))
    
    if not yaml_files:
        print(f"❌ No YAML files found in {config_dir}")
        return
    
    print(f"📊 Generating visualizations for {len(yaml_files)} configurations...")
    
    for yaml_file in yaml_files:
        output_file = output_dir / f"{yaml_file.stem}_visualization.png"
        try:
            visualize_config(yaml_file, output_file)
            print(f"  ✅ {yaml_file.name} → {output_file.name}")
        except Exception as e:
            print(f"  ❌ {yaml_file.name}: {e}")
    
    print(f"\n✅ All visualizations saved to {output_dir}/")
    

def generate_beginner_guide(config_path, output_path="beginner_guide.txt"):
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    devices = config.get('devices', [])
    topology = config.get('topology', [])
    tasks = config.get('tasks', [])
    dependencies = config.get('dependencies', [])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BEGINNER'S GUIDE TO HPC CONFIGURATION\n")
        f.write("BY Dr. Patrick Lemoine\n")
        f.write(f"Configuration: {config_path.name}\n")
        f.write("=" * 80 + "\n\n")
        
        # Introduction
        f.write("📖 WHAT IS THIS?\n")
        f.write("-" * 80 + "\n")
        f.write("This configuration describes a High-Performance Computing (HPC) system.\n")
        f.write("It defines:\n")
        f.write("  • Hardware devices (CPUs, GPUs, accelerators)\n")
        f.write("  • Network connections between devices\n")
        f.write("  • Tasks to be executed\n")
        f.write("  • Dependencies between tasks\n\n")
        
        # Devices explanation
        f.write("=" * 80 + "\n")
        f.write("🖥️  DEVICES (HARDWARE COMPONENTS)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total devices: {len(devices)}\n\n")
        
        if devices:
            f.write("WHAT ARE DEVICES?\n")
            f.write("-" * 80 + "\n")
            f.write("Devices are processing units that execute computations.\n")
            f.write("Different types are optimized for different workloads:\n\n")
            
            f.write("  • CPU (Central Processing Unit):\n")
            f.write("    - General-purpose processor\n")
            f.write("    - Good for: Sequential tasks, system control, data preprocessing\n")
            f.write("    - Typical speed: 100-200 TFLOPS\n\n")
            
            f.write("  • GPU (Graphics Processing Unit):\n")
            f.write("    - Massively parallel processor\n")
            f.write("    - Good for: Deep learning training, matrix operations, simulations\n")
            f.write("    - Typical speed: 1000-2000 TFLOPS\n\n")
            
            f.write("  • NPU (Neural Processing Unit):\n")
            f.write("    - Specialized for AI inference\n")
            f.write("    - Good for: Real-time AI predictions, edge computing\n")
            f.write("    - Typical speed: 500-1000 TFLOPS\n\n")
            
            f.write("  • TPU (Tensor Processing Unit):\n")
            f.write("    - Optimized for tensor operations\n")
            f.write("    - Good for: Large-scale ML training, tensor computations\n")
            f.write("    - Typical speed: 800-1200 TFLOPS\n\n")
            
            f.write("  • DPU (Data Processing Unit):\n")
            f.write("    - Specialized for data movement and I/O\n")
            f.write("    - Good for: Network processing, storage acceleration\n")
            f.write("    - Typical speed: 200-300 TFLOPS\n\n")
            
            f.write("\nYOUR DEVICES:\n")
            f.write("-" * 80 + "\n")
            
            device_by_kind = {}
            for dev in devices:
                kind = dev.get('kind', 'unknown')
                if kind not in device_by_kind:
                    device_by_kind[kind] = []
                device_by_kind[kind].append(dev)
            
            for kind, devs in sorted(device_by_kind.items()):
                f.write(f"\n{kind.upper()} ({len(devs)} device{'s' if len(devs) > 1 else ''}):\n")
                for dev in devs:
                    f.write(f"  • {dev['name']}:\n")
                    f.write(f"    - Speed: {dev.get('speed', 0)} TFLOPS\n")
                    f.write(f"    - Threads: {dev.get('threads', 0)} parallel execution units\n")
                    
                    memories = dev.get('memories', [])
                    if memories:
                        f.write(f"    - Memory hierarchy:\n")
                        for mem in memories:
                            cap_gb = mem.get('capacity', 0) / 1024
                            f.write(f"      * {mem.get('name', 'MEM')}: {cap_gb:.1f} GB @ {mem.get('bandwidth', 0)} GB/s\n")
                            f.write(f"        (Lower latency = faster access)\n")
        
        # Topology explanation
        f.write("\n" + "=" * 80 + "\n")
        f.write("🔗 NETWORK TOPOLOGY (DEVICE CONNECTIONS)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("WHAT IS TOPOLOGY?\n")
        f.write("-" * 80 + "\n")
        f.write("Topology describes how devices are connected and communicate.\n")
        f.write("Key metrics:\n\n")
        
        f.write("  • Bandwidth (GB/s):\n")
        f.write("    - How much data can be transferred per second\n")
        f.write("    - Higher = faster data movement\n")
        f.write("    - Example: 100 GB/s can transfer 100 GB in 1 second\n\n")
        
        f.write("  • Latency (nanoseconds):\n")
        f.write("    - Time delay before transfer begins\n")
        f.write("    - Lower = faster response\n")
        f.write("    - Example: 5 ns latency means 5 billionths of a second delay\n\n")
        
        f.write("Connection types:\n")
        f.write("  • XGMI (high-speed GPU interconnect): 500-2000 GB/s, 1-3 ns latency\n")
        f.write("  • PCIe (standard device connection): 50-150 GB/s, 5-10 ns latency\n")
        f.write("  • Fabric (network fabric): 100-500 GB/s, 3-8 ns latency\n\n")
        
        if topology:
            f.write(f"YOUR NETWORK ({len(topology)} connections):\n")
            f.write("-" * 80 + "\n")
            
            # Group by connection type
            links_by_kind = {}
            for link in topology:
                kind = link[4] if len(link) > 4 else 'unknown'
                if kind not in links_by_kind:
                    links_by_kind[kind] = []
                links_by_kind[kind].append(link)
            
            for kind, links in sorted(links_by_kind.items()):
                f.write(f"\n{kind.upper()} connections ({len(links)}):\n")
                # Show statistics
                bandwidths = [l[2] for l in links if len(l) > 2]
                latencies = [l[3] for l in links if len(l) > 3]
                
                if bandwidths:
                    avg_bw = sum(bandwidths) / len(bandwidths)
                    f.write(f"  Average bandwidth: {avg_bw:.1f} GB/s\n")
                if latencies:
                    avg_lat = sum(latencies) / len(latencies)
                    f.write(f"  Average latency: {avg_lat:.1f} ns\n")
                
                f.write(f"  Example connections:\n")
                for link in links[:3]:
                    src, dst = link[0], link[1]
                    bw = link[2] if len(link) > 2 else 0
                    lat = link[3] if len(link) > 3 else 0
                    f.write(f"    {src} ↔ {dst}: {bw} GB/s, {lat} ns latency\n")
                
                if len(links) > 3:
                    f.write(f"    ... and {len(links) - 3} more\n")
        
        # Tasks explanation
        f.write("\n" + "=" * 80 + "\n")
        f.write("📋 TASKS (COMPUTATIONAL WORKLOADS)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("WHAT ARE TASKS?\n")
        f.write("-" * 80 + "\n")
        f.write("Tasks are units of work to be executed on devices.\n")
        f.write("Each task has:\n\n")
        
        f.write("  • Compute (TFLOPS):\n")
        f.write("    - Amount of computation required\n")
        f.write("    - Higher = more processing time needed\n\n")
        
        f.write("  • Memory (MB):\n")
        f.write("    - Amount of RAM/memory required\n")
        f.write("    - Task won't run if device doesn't have enough memory\n\n")
        
        f.write("  • Priority (1-10):\n")
        f.write("    - Importance of the task\n")
        f.write("    - 10 = highest priority (most urgent)\n")
        f.write("    - 1 = lowest priority\n\n")
        
        f.write("  • Deadline (milliseconds):\n")
        f.write("    - Time by which task must complete\n")
        f.write("    - Exceeding deadline may cause system failure or poor performance\n\n")
        
        f.write("  • Preferred device kind:\n")
        f.write("    - Optimal device type for this task\n")
        f.write("    - Scheduler tries to place task on preferred device\n\n")
        
        if tasks:
            f.write(f"YOUR TASKS ({len(tasks)} total):\n")
            f.write("-" * 80 + "\n\n")
            
            total_compute = sum(t.get('compute', 0) for t in tasks)
            total_memory = sum(t.get('memory', 0) for t in tasks)
            avg_priority = sum(t.get('priority', 0) for t in tasks) / len(tasks)
            avg_deadline = sum(t.get('deadline', 0) for t in tasks) / len(tasks)
            
            f.write("OVERALL STATISTICS:\n")
            f.write(f"  Total compute required: {total_compute} TFLOPS\n")
            f.write(f"  Total memory required: {total_memory / 1024:.1f} GB\n")
            f.write(f"  Average priority: {avg_priority:.1f}/10\n")
            f.write(f"  Average deadline: {avg_deadline:.0f} ms\n\n")
            
            # Group by preferred device
            tasks_by_kind = {}
            for task in tasks:
                kind = task.get('preferred_kind', 'any')
                if kind not in tasks_by_kind:
                    tasks_by_kind[kind] = []
                tasks_by_kind[kind].append(task)
            
            for kind, task_list in sorted(tasks_by_kind.items()):
                f.write(f"\nTasks preferring {kind.upper()} ({len(task_list)}):\n")
                for task in task_list:
                    f.write(f"  • {task['name']}:\n")
                    f.write(f"    - Compute: {task.get('compute', 0)} TFLOPS\n")
                    f.write(f"    - Memory: {task.get('memory', 0) / 1024:.2f} GB\n")
                    f.write(f"    - Priority: {task.get('priority', 0)}/10\n")
                    f.write(f"    - Deadline: {task.get('deadline', 0)} ms\n")
                    
                    # Interpretation
                    priority = task.get('priority', 0)
                    if priority >= 8:
                        f.write(f"    → HIGH PRIORITY: Must complete quickly\n")
                    elif priority <= 3:
                        f.write(f"    → LOW PRIORITY: Can be delayed if needed\n")
        
        # Dependencies explanation
        f.write("\n" + "=" * 80 + "\n")
        f.write("🔀 TASK DEPENDENCIES (EXECUTION ORDER)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("WHAT ARE DEPENDENCIES?\n")
        f.write("-" * 80 + "\n")
        f.write("Dependencies define the order in which tasks must execute.\n")
        f.write("A dependency 'A → B' means:\n")
        f.write("  • Task A must complete BEFORE task B can start\n")
        f.write("  • Data produced by A is needed by B\n")
        f.write("  • The amount of data transferred is specified in MB\n\n")
        
        if dependencies:
            f.write(f"YOUR DEPENDENCIES ({len(dependencies)}):\n")
            f.write("-" * 80 + "\n\n")
            
            total_data_transfer = sum(d.get('bytes', 0) for d in dependencies)
            f.write(f"Total data to transfer: {total_data_transfer} MB\n\n")
            
            f.write("Dependency chain:\n")
            for dep in dependencies:
                src = dep['src']
                dst = dep['dst']
                data = dep.get('bytes', 0)
                f.write(f"  {src} → {dst} ({data} MB of data)\n")
            
            f.write("\nIMPLICATIONS:\n")
            f.write("  • Tasks cannot run in parallel if one depends on another\n")
            f.write("  • Large data transfers can become bottlenecks\n")
            f.write("  • Network bandwidth affects overall completion time\n")
        
        # How to use this
        f.write("\n" + "=" * 80 + "\n")
        f.write("🎯 HOW TO USE THIS CONFIGURATION\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. TRAINING THE AI SCHEDULER:\n")
        f.write("   python main.py train --config configs/your_config.yaml\n")
        f.write("   → The AI learns to assign tasks to devices optimally\n\n")
        
        f.write("2. EVALUATING PERFORMANCE:\n")
        f.write("   python main.py evaluate --config configs/your_config.yaml\n")
        f.write("   → Tests how well the AI scheduler performs\n\n")
        
        f.write("3. BENCHMARKING:\n")
        f.write("   python main.py benchmark --config configs/your_config.yaml --episodes 20\n")
        f.write("   → Runs multiple tests to get average performance\n\n")
        
        f.write("4. VISUALIZING THE SYSTEM:\n")
        f.write("   python main.py visualize --config configs/your_config.yaml\n")
        f.write("   → Creates a diagram showing devices, network, and tasks\n\n")
        
        f.write("5. COMPARING CONFIGURATIONS:\n")
        f.write("   python main.py compare --configs config1.yaml config2.yaml\n")
        f.write("   → Shows differences between multiple configurations\n\n")
        
        # Understanding results
        f.write("\n" + "=" * 80 + "\n")
        f.write("📊 UNDERSTANDING THE RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("After running evaluation or benchmark, you'll see metrics:\n\n")
        
        f.write("• total_reward:\n")
        f.write("  - Overall performance score (higher is better)\n")
        f.write("  - Combines efficiency, task completion, and deadline adherence\n\n")
        
        f.write("• late_tasks:\n")
        f.write("  - Number of tasks that missed their deadline\n")
        f.write("  - Lower is better (0 = all tasks completed on time)\n\n")
        
        f.write("• congestion:\n")
        f.write("  - How many devices are overloaded\n")
        f.write("  - Lower is better (indicates good load balancing)\n\n")
        
        f.write("• migrations:\n")
        f.write("  - Number of times tasks were moved between devices\n")
        f.write("  - Too many = inefficient (wastes time on data transfer)\n")
        f.write("  - Too few = inflexible (can't adapt to changes)\n\n")
        
        f.write("• segments:\n")
        f.write("  - Number of task execution segments\n")
        f.write("  - Shows how work is divided across time and devices\n\n")
        
        # Tips for beginners
        f.write("\n" + "=" * 80 + "\n")
        f.write("💡 TIPS FOR BEGINNERS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. START SMALL:\n")
        f.write("   - Begin with small_cluster.yaml (2 devices, 3 tasks)\n")
        f.write("   - Understand the basics before moving to complex configs\n\n")
        
        f.write("2. CHECK DEVICE CAPACITY:\n")
        f.write("   - Make sure devices have enough memory for tasks\n")
        f.write("   - Sum of task memory should not exceed device memory\n\n")
        
        f.write("3. BALANCE PRIORITIES:\n")
        f.write("   - Not all tasks should be priority 10\n")
        f.write("   - Mix of priorities allows flexible scheduling\n\n")
        
        f.write("4. REALISTIC DEADLINES:\n")
        f.write("   - Set deadlines based on task compute requirements\n")
        f.write("   - Too tight = many late tasks\n")
        f.write("   - Too loose = inefficient resource use\n\n")
        
        f.write("5. NETWORK MATTERS:\n")
        f.write("   - Fast connections (XGMI) allow efficient task migration\n")
        f.write("   - Slow connections create bottlenecks\n\n")
        
        f.write("6. MONITOR METRICS:\n")
        f.write("   - Watch late_tasks and congestion closely\n")
        f.write("   - These indicate scheduling quality\n\n")
        
        f.write("7. ITERATE:\n")
        f.write("   - Try different configurations\n")
        f.write("   - Compare results to find optimal setup\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF GUIDE\n")
        f.write("=" * 80 + "\n")
    
    print(f"✅ Beginner's guide saved to: {output_path}")