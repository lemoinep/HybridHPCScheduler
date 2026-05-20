import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def export_segment_gantt(segments, filename="output/gantt_segments.png"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 7))
    segments = sorted(segments, key=lambda s: (s.start, s.task, s.segment_id))
    colors = plt.cm.tab20(np.linspace(0, 1, len(segments)))

    for i, seg in enumerate(segments):
        label = f"{seg.task}:{seg.segment_id}"
        ax.barh(label, seg.duration, left=seg.start, color=colors[i], edgecolor="black")
        if seg.migrated:
            ax.text(seg.start + seg.duration / 2, i, "M", ha="center", va="center", color="white", fontsize=8)

    ax.set_xlabel("Time")
    ax.set_ylabel("Task segments")
    ax.set_title("Segment-aware APIC schedule")
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(filename, dpi=180)
    plt.close(fig)


def export_network(graph, filename="output/network.png"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(graph, seed=3)
    nx.draw_networkx_nodes(graph, pos, node_size=1200, node_color="#4169e1", ax=ax)
    nx.draw_networkx_labels(graph, pos, font_color="white", ax=ax)
    nx.draw_networkx_edges(graph, pos, arrows=True, alpha=0.35, ax=ax)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=180)
    plt.close(fig)