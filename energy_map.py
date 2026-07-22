import argparse
import csv
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import yaml

import subprocess
import pynvml
import re


import re

def _get_cpu_vendor_proc():
    try:
        with open("/proc/cpuinfo", "r") as f:
            content = f.read()
    except OSError:
        return None

    match = re.search(r"^vendor_id\s+:\s+(.+)$", content, re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip()


def is_intel():
    vendor = _get_cpu_vendor_proc()
    return vendor == "GenuineIntel"


def is_amd():
    vendor = _get_cpu_vendor_proc()
    return vendor == "AuthenticAMD"




def load_topology(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def read_rapl_packages(energy_cfg):
    amd_cfg = energy_cfg.get("amd", {})
    amd_mode = amd_cfg.get("mode", "real")
    amd_packages = int(amd_cfg.get("packages", 2))
    amd_interval = float(amd_cfg.get("interval_seconds", 0.1))

    if is_intel():
        cmd = ["./read_rapl_packages_intel"]
    elif is_amd():
        if amd_mode == "simulated":
            cmd = ["./read_rapl_packages_amd_stub", str(amd_packages)]
        else:
            cmd = ["./read_rapl_packages_amd_real", str(amd_interval), str(amd_packages)]
    else:
        raise RuntimeError("Unsupported CPU vendor for RAPL")

    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = proc.stdout.strip().splitlines()
    reader = csv.DictReader(lines)
    powers = {}
    for row in reader:
        pkg_id = int(row["package_id"])
        powers[pkg_id] = float(row["power_watts"])
    return powers

def read_cpu_metrics_from_topology(topology, rapl_powers):
    cpu_rows = []
    for cpu in topology.get("cpus", []):
        pkg = cpu.get("socket", 0)
        power = rapl_powers.get(pkg, 0.0)

        temp = random.uniform(50, 80)
        cpu_rows.append({
            "device_id": cpu["id"],
            "device_type": "CPU",
            "temperature": temp,
            "power_watts": power,
        })
    return cpu_rows

def read_gpu_metrics_from_topology(topology):
    pynvml.nvmlInit()
    gpu_rows = []
    gpus = topology.get("gpus", [])

    for i, gpu in enumerate(gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        power_w = power_mw / 1000.0
        gpu_rows.append({
            "device_id": gpu["id"],
            "device_type": "GPU",
            "temperature": float(temp),
            "power_watts": float(power_w),
        })
    pynvml.nvmlShutdown()
    return gpu_rows

def generate_real_metrics(devices, cfg, output_dir, topology_path):
    topology = load_topology(topology_path)

    # 1. CPU via RAPL
    rapl_powers = read_rapl_packages()
    cpu_rows = read_cpu_metrics_from_topology(topology, rapl_powers)

    # 2. GPU via NVML
    gpu_rows = read_gpu_metrics_from_topology(topology)

    # 3. NPU/TPU/DPU 
    sim_cfg = cfg["simulation"]
    extra_rows = []
    for dev in devices:
        if dev["type"] in ("TPU", "NPU", "DPU"):
            p = sim_cfg[dev["type"].lower()]
            temp = random.uniform(p["temp_min"], p["temp_max"])
            power = random.uniform(p["power_min"], p["power_max"])
            extra_rows.append({
                "device_id": dev["id"],
                "device_type": dev["type"],
                "temperature": temp,
                "power_watts": power,
            })

    rows = cpu_rows + gpu_rows + extra_rows

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "device_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["device_id", "device_type", "temperature", "power_watts"],
        )
        writer.writeheader()
        writer.writerows(rows)


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_devices_from_energy_config(cfg):
    devices_cfg = cfg["devices"]
    devices = []

    for i in range(devices_cfg.get("cpu_count", 0)):
        devices.append({"id": f"CPU{i}", "type": "CPU"})

    for i in range(devices_cfg.get("gpu_count", 0)):
        devices.append({"id": f"GPU{i}", "type": "GPU"})

    for i in range(devices_cfg.get("tpu_count", 0)):
        devices.append({"id": f"TPU{i}", "type": "TPU"})

    for i in range(devices_cfg.get("npu_count", 0)):
        devices.append({"id": f"NPU{i}", "type": "NPU"})

    for i in range(devices_cfg.get("dpu_count", 0)):
        devices.append({"id": f"DPU{i}", "type": "DPU"})

    return devices


def generate_simulated_metrics(devices, cfg, output_dir):
    sim_cfg = cfg["simulation"]
    rows = []

    type_map = {
        "CPU": "cpu",
        "GPU": "gpu",
        "TPU": "tpu",
        "NPU": "npu",
        "DPU": "dpu",
    }

    for dev in devices:
        key = type_map[dev["type"]]
        p = sim_cfg[key]
        temp = random.uniform(p["temp_min"], p["temp_max"])
        power = random.uniform(p["power_min"], p["power_max"])
        rows.append({
            "device_id": dev["id"],
            "device_type": dev["type"],
            "temperature": temp,
            "power_watts": power,
        })

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "device_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["device_id", "device_type", "temperature", "power_watts"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Simulated metrics written to {csv_path}")


def load_device_metrics(csv_path):
    metrics = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics[row["device_id"]] = {
                "type": row["device_type"],
                "temperature": float(row["temperature"]),
                "power_watts": float(row["power_watts"]),
            }
    return metrics


def build_matrix(devices, metrics, rows, cols, value_key):
    matrix = np.zeros((rows, cols))
    labels = [[None for _ in range(cols)] for _ in range(rows)]

    for idx, dev in enumerate(devices):
        if idx >= rows * cols:
            break
        r = idx // cols
        c = idx % cols
        val = metrics[dev["id"]][value_key]
        matrix[r, c] = val
        labels[r][c] = dev["id"]

    return matrix, labels


def plot_energy_maps(devices, metrics, cfg, output_dir):
    graphics_cfg = cfg["graphics"]
    rows = graphics_cfg["rows"]
    cols = graphics_cfg["cols"]
    cmap_name = graphics_cfg["colormap"]
    show_labels = graphics_cfg.get("show_labels", True)

    # Power map
    power_matrix, labels = build_matrix(devices, metrics, rows, cols, "power_watts")
    plt.figure(figsize=(10, 7))
    plt.imshow(power_matrix, cmap=cmap_name)
    plt.colorbar(label="Power (W)")
    if show_labels:
        for r in range(rows):
            for c in range(cols):
                if labels[r][c] is not None:
                    plt.text(
                        c,
                        r,
                        labels[r][c],
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white",
                    )
    plt.title("Energy map (power)")
    plt.tight_layout()
    power_path = os.path.join(output_dir, "energy_map_power.png")
    plt.savefig(power_path, dpi=200)
    plt.close()
    print(f"[INFO] Power map written to {power_path}")

    # Temperature map
    temp_matrix, labels = build_matrix(devices, metrics, rows, cols, "temperature")
    plt.figure(figsize=(10, 7))
    plt.imshow(temp_matrix, cmap=cmap_name)
    plt.colorbar(label="Temperature (°C)")
    if show_labels:
        for r in range(rows):
            for c in range(cols):
                if labels[r][c] is not None:
                    plt.text(
                        c,
                        r,
                        labels[r][c],
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white",
                    )
    plt.title("Energy map (temperature)")
    plt.tight_layout()
    temp_path = os.path.join(output_dir, "energy_map_temperature.png")
    plt.savefig(temp_path, dpi=200)
    plt.close()
    print(f"[INFO] Temperature map written to {temp_path}")



def main():
    parser = argparse.ArgumentParser(
        description="Generate energy/thermal maps (simulated or real)"
    )
    parser.add_argument(
        "--energy-config",
        required=True,
        help="YAML file for energy simulation/graphics",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for maps and metrics",
    )
    parser.add_argument(
        "--mode",
        choices=["simulated", "real"],
        default="simulated",
        help="Run in simulated or real metrics mode",
    )
    
    parser.add_argument(
        "--topology",
        required=False,
        help="YAML topology generated by scan_exascale_hwloc (for real mode)",
    )

    args = parser.parse_args()

    energy_cfg = load_yaml(args.energy_config)
    devices = build_devices_from_energy_config(energy_cfg)

    if args.mode == "simulated":
        generate_simulated_metrics(devices, energy_cfg, args.output)
    
    if args.mode == "real":
        if not args.topology:
            parser.error("--topology is required in real mode")
        generate_real_metrics(devices, energy_cfg, args.output, args.topology)
    
    metrics = load_device_metrics(os.path.join(args.output, "device_metrics.csv"))
    plot_energy_maps(devices, metrics, energy_cfg, args.output)


if __name__ == "__main__":
    main()
    
