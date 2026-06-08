#define NOMINMAX

// Author(s): Dr. Patrick Lemoine
// Version 1.3 (Linux) 06/06/2026 — CPU = PUs, real GPU detection via hwloc
// Note: NPU, TPU, DPU are synthetic accelerators controlled by configuration or cluster description.

#include <hwloc.h>
#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cstring>

struct SimConfig {
    int gpu_count_sim = 0;
    int npu_count = 1;
    int tpu_count = 1;
    int dpu_count = 1;

    int cpu_speed = 120;
    int cpu_threads_per_socket = -1; // unused in PU mode
    int cpu_dram_capacity = 131072;
    int cpu_dram_latency = 80;
    int cpu_dram_bandwidth = 200;
    int cpu_l3_capacity = 32768;
    int cpu_l3_latency = 12;
    int cpu_l3_bandwidth = 800;

    int gpu_speed = 1200;
    int gpu_threads = 256;
    int gpu_mem_capacity = 49152;
    int gpu_mem_latency = 3;
    int gpu_mem_bandwidth = 1600;
    int gpu_l2_capacity = 8192;
    int gpu_l2_latency = 1;
    int gpu_l2_bandwidth = 3000;

    int npu_speed = 800;
    int npu_threads = 128;
    int npu_mem_capacity = 16384;
    int npu_mem_latency = 2;
    int npu_mem_bandwidth = 1400;
    int npu_cache_capacity = 4096;
    int npu_cache_latency = 1;
    int npu_cache_bandwidth = 2500;

    int tpu_speed = 950;
    int tpu_threads = 128;
    int tpu_mem_capacity = 32768;
    int tpu_mem_latency = 3;
    int tpu_mem_bandwidth = 1500;
    int tpu_cache_capacity = 4096;
    int tpu_cache_latency = 1;
    int tpu_cache_bandwidth = 2500;

    int dpu_speed = 250;
    int dpu_threads = 32;
    int dpu_mem_capacity = 8192;
    int dpu_mem_latency = 2;
    int dpu_mem_bandwidth = 1000;
    int dpu_cache_capacity = 2048;
    int dpu_cache_latency = 1;
    int dpu_cache_bandwidth = 1500;

    int task_priority_base = 3;
    int task_deadline_base = 4000;

    bool detect_cpu_real = true;
    bool detect_gpu_real = true;    // enabled on Linux
    bool add_simulated_gpu = true;

    bool real_only = false;         // real hardware only
    bool enable_synthetic_accels = true; // synthetic NPU/TPU/DPU

    bool verbose = false;           // verbose logging

    std::string node_type;          // logical node type from cluster description
    std::string cluster_desc = "ClusterDescription.yaml";
};

struct DeviceInfo {
    std::string name;
    std::string kind;
    int speed;
    int threads;
    int dram_cap = 0;
    int dram_lat = 0;
    int dram_bw = 0;
    int cache_cap = 0;
    int cache_lat = 0;
    int cache_bw = 0;
    bool is_real = false;
};

static int nbobjs(hwloc_topology_t topo, hwloc_obj_type_t type) {
    int n = hwloc_get_nbobjs_by_type(topo, type);
    return n < 0 ? 0 : n;
}

static YAML::Node memnode(const std::string& name, int cap, int lat, int bw) {
    YAML::Node m;
    m["name"] = name;
    m["capacity"] = cap;
    m["latency"] = lat;
    m["bandwidth"] = bw;
    return m;
}

static YAML::Node device_node(const std::string& name, const std::string& kind, int speed, int threads,
                              int mem_cap, int mem_lat, int mem_bw,
                              int cache_cap, int cache_lat, int cache_bw) {
    YAML::Node d;
    d["name"] = name;
    d["kind"] = kind;
    d["speed"] = speed;
    d["threads"] = threads;
    YAML::Node mems = YAML::Node(YAML::NodeType::Sequence);
    mems.push_back(memnode(kind == "cpu" ? "DRAM" : (kind == "gpu" ? "HBM" : "SRAM"), mem_cap, mem_lat, mem_bw));
    mems.push_back(memnode(kind == "cpu" ? "L3" : (kind == "gpu" ? "L2" : "CACHE"), cache_cap, cache_lat, cache_bw));
    d["memories"] = mems;
    return d;
}

static YAML::Node link_node(const std::string& a, const std::string& b, int speed, int latency, const std::string& type) {
    YAML::Node n = YAML::Node(YAML::NodeType::Sequence);
    n.push_back(a);
    n.push_back(b);
    n.push_back(speed);
    n.push_back(latency);
    n.push_back(type);
    return n;
}

static void add_arg_int(int argc, char** argv, const std::string& key, int& target) {
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s.rfind(key + "=", 0) == 0) {
            target = std::stoi(s.substr(key.size() + 1));
        }
    }
}

static void add_arg_bool(int argc, char** argv, const std::string& key, bool& target) {
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == key) target = true;
        if (s.rfind(key + "=", 0) == 0) {
            target = (s.substr(key.size() + 1) == "true");
        }
    }
}

static std::string cpu_name(int i) { return "CPU" + std::to_string(i); }
static std::string gpu_name(int i) { return "GPU" + std::to_string(i); }
static std::string npu_name(int i) { return "NPU" + std::to_string(i); }
static std::string tpu_name(int i) { return "TPU" + std::to_string(i); }
static std::string dpu_name(int i) { return "DPU" + std::to_string(i); }

int main(int argc, char** argv) {
    SimConfig cfg;

    // Parse integer arguments
    add_arg_int(argc, argv, "--gpu-count-sim", cfg.gpu_count_sim);
    add_arg_int(argc, argv, "--npu-count",     cfg.npu_count);
    add_arg_int(argc, argv, "--tpu-count",     cfg.tpu_count);
    add_arg_int(argc, argv, "--dpu-count",     cfg.dpu_count);

    add_arg_int(argc, argv, "--cpu-speed", cfg.cpu_speed);
    add_arg_int(argc, argv, "--gpu-speed", cfg.gpu_speed);
    add_arg_int(argc, argv, "--npu-speed", cfg.npu_speed);
    add_arg_int(argc, argv, "--tpu-speed", cfg.tpu_speed);
    add_arg_int(argc, argv, "--dpu-speed", cfg.dpu_speed);

    add_arg_int(argc, argv, "--task-priority-base", cfg.task_priority_base);
    add_arg_int(argc, argv, "--task-deadline-base", cfg.task_deadline_base);

    // Boolean flags with correct semantics
    bool no_real_cpu = false;
    bool no_real_gpu = false;
    bool no_sim_gpu  = false;

    add_arg_bool(argc, argv, "--no-real-cpu", no_real_cpu);
    add_arg_bool(argc, argv, "--no-real-gpu", no_real_gpu);
    add_arg_bool(argc, argv, "--no-sim-gpu",  no_sim_gpu);
    add_arg_bool(argc, argv, "--real",        cfg.real_only);
    add_arg_bool(argc, argv, "--verbose",     cfg.verbose);

    // Parse node_type and cluster_desc
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s.rfind("--node-type=", 0) == 0) {
            cfg.node_type = s.substr(std::string("--node-type=").size());
        } else if (s.rfind("--cluster-desc=", 0) == 0) {
            cfg.cluster_desc = s.substr(std::string("--cluster-desc=").size());
        }
    }

    // Apply boolean flags
    if (no_real_cpu) cfg.detect_cpu_real = false;
    if (no_real_gpu) cfg.detect_gpu_real = false;
    if (no_sim_gpu)  cfg.add_simulated_gpu = false;

    // Real-only mode: no simulated accelerators at all
    if (cfg.real_only) {
        cfg.add_simulated_gpu       = false;
        cfg.enable_synthetic_accels = false;
    }

    // If user sets all synthetic counts to 0, disable them explicitly
    if (cfg.npu_count == 0 && cfg.tpu_count == 0 && cfg.dpu_count == 0) {
        cfg.enable_synthetic_accels = false;
    }

    std::string out = "ScanExascale.yaml";
    std::cout << "Save File\n" << out << "\n";

    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s.rfind("--output=", 0) == 0) out = s.substr(9);
    }

    hwloc_topology_t topo;
    if (hwloc_topology_init(&topo) != 0) {
        std::cerr << "Failed to init hwloc topology\n";
        return 1;
    }

    // On Linux, enable I/O device discovery so we see GPUs, etc.
    unsigned io_flags = HWLOC_TOPOLOGY_FLAG_NO_DISTANCES | HWLOC_TOPOLOGY_FLAG_NO_MEMATTRS;
    hwloc_topology_set_flags(topo, io_flags);

    if (hwloc_topology_load(topo) != 0) {
        std::cerr << "Failed to load hwloc topology\n";
        hwloc_topology_destroy(topo);
        return 1;
    }

    // PUs = logical processors (Linux CPU 0..N)
    int pus  = nbobjs(topo, HWLOC_OBJ_PU);

    // If node_type is defined, we may override PUs and accelerator counts from cluster description
    if (!cfg.node_type.empty()) {
        try {
            YAML::Node cluster = YAML::LoadFile(cfg.cluster_desc);
            if (cluster["node_types"]) {
                for (const auto& nt : cluster["node_types"]) {
                    if (!nt["name"] || nt["name"].as<std::string>() != cfg.node_type)
                        continue;

                    if (nt["cpu"]) {
                        if (nt["cpu"]["pus"])
                            pus = nt["cpu"]["pus"].as<int>();
                        if (nt["cpu"]["speed"])
                            cfg.cpu_speed = nt["cpu"]["speed"].as<int>();
                    }

                    if (nt["gpu"]) {
                        if (nt["gpu"]["count"])
                            cfg.gpu_count_sim = nt["gpu"]["count"].as<int>();
                        if (nt["gpu"]["speed"])
                            cfg.gpu_speed = nt["gpu"]["speed"].as<int>();
                        if (nt["gpu"]["threads"])
                            cfg.gpu_threads = nt["gpu"]["threads"].as<int>();
                        if (nt["gpu"]["mem_capacity"])
                            cfg.gpu_mem_capacity = nt["gpu"]["mem_capacity"].as<int>();
                        if (nt["gpu"]["mem_bandwidth"])
                            cfg.gpu_mem_bandwidth = nt["gpu"]["mem_bandwidth"].as<int>();
                    }

                    if (nt["npu"] && nt["npu"]["count"])
                        cfg.npu_count = nt["npu"]["count"].as<int>();
                    if (nt["tpu"] && nt["tpu"]["count"])
                        cfg.tpu_count = nt["tpu"]["count"].as<int>();
                    if (nt["dpu"] && nt["dpu"]["count"])
                        cfg.dpu_count = nt["dpu"]["count"].as<int>();

                    break;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: failed to load cluster description from "
                      << cfg.cluster_desc << ": " << e.what() << "\n";
        }
    }

    int cpus = std::max(1, pus);

    if (cfg.verbose) {
        std::cerr << "=== hwloc topology summary ===\n";
        std::cerr << "PUs (logical processors): " << pus << "\n";

        std::cerr << "=== hwloc OS devices ===\n";
        hwloc_obj_t osdev_dbg = nullptr;
        while ((osdev_dbg = hwloc_get_next_obj_by_type(topo, HWLOC_OBJ_OS_DEVICE, osdev_dbg)) != nullptr) {
            if (!osdev_dbg->attr) continue;
            auto type = osdev_dbg->attr->osdev.type;
            const char* name    = osdev_dbg->name ? osdev_dbg->name : "NULL";
            const char* subtype = osdev_dbg->subtype ? osdev_dbg->subtype : "";

            std::cerr << "OS device: name=" << name
                      << " type=" << type
                      << " subtype=" << subtype
                      << "\n";
        }
        std::cerr << "==============================\n";
    }

    cfg.cpu_threads_per_socket = 1;

    std::vector<DeviceInfo> devices;
    int gpu_idx = 0;
    int real_gpu_count = 0;

    // Real CPUs from hwloc PUs (or overridden PUs)
    if (cfg.detect_cpu_real) {
        for (int i = 0; i < cpus; ++i) {
            DeviceInfo d;
            d.name       = cpu_name(i);
            d.kind       = "cpu";
            d.speed      = cfg.cpu_speed;
            d.threads    = cfg.cpu_threads_per_socket;
            d.dram_cap   = cfg.cpu_dram_capacity;
            d.dram_lat   = cfg.cpu_dram_latency;
            d.dram_bw    = cfg.cpu_dram_bandwidth;
            d.cache_cap  = cfg.cpu_l3_capacity;
            d.cache_lat  = cfg.cpu_l3_latency;
            d.cache_bw   = cfg.cpu_l3_bandwidth;
            d.is_real    = true;
            devices.push_back(d);
        }
    }

    // Real GPUs via OS devices (requires hwloc built with GPU backends)
    if (cfg.detect_gpu_real && cfg.node_type.empty()) {
        // In node_type mode we typically want synthetic counts, so only probe when node_type is empty
        hwloc_obj_t osdev = nullptr;
        while ((osdev = hwloc_get_next_obj_by_type(topo, HWLOC_OBJ_OS_DEVICE, osdev)) != nullptr) {
            if (!osdev->attr) continue;
            hwloc_obj_osdev_type_e type = osdev->attr->osdev.type;

            if (type == HWLOC_OBJ_OSDEV_GPU || (osdev->attr->osdev.type & HWLOC_OBJ_OSDEV_GPU)) {
                DeviceInfo d;
                d.name       = "GPU" + std::to_string(real_gpu_count);
                d.kind       = "gpu";
                d.speed      = cfg.gpu_speed;
                d.threads    = cfg.gpu_threads;
                d.dram_cap   = cfg.gpu_mem_capacity;
                d.dram_lat   = cfg.gpu_mem_latency;
                d.dram_bw    = cfg.gpu_mem_bandwidth;
                d.cache_cap  = cfg.gpu_l2_capacity;
                d.cache_lat  = cfg.gpu_l2_latency;
                d.cache_bw   = cfg.gpu_l2_bandwidth;
                d.is_real    = true;
                devices.push_back(d);
                ++real_gpu_count;
            }
        }
        gpu_idx = real_gpu_count;
    }

    // Simulated GPUs (only if not in real-only mode and flag enabled)
    if (cfg.add_simulated_gpu && !cfg.real_only) {
        int target_gpus = (cfg.gpu_count_sim > 0) ? cfg.gpu_count_sim : 2;

        while (gpu_idx < target_gpus) {
            DeviceInfo d;
            d.name       = gpu_name(gpu_idx++);
            d.kind       = "gpu";
            d.speed      = cfg.gpu_speed;
            d.threads    = cfg.gpu_threads;
            d.dram_cap   = cfg.gpu_mem_capacity;
            d.dram_lat   = cfg.gpu_mem_latency;
            d.dram_bw    = cfg.gpu_mem_bandwidth;
            d.cache_cap  = cfg.gpu_l2_capacity;
            d.cache_lat  = cfg.gpu_l2_latency;
            d.cache_bw   = cfg.gpu_l2_bandwidth;
            d.is_real    = false;
            devices.push_back(d);
        }
    }

    // Simulated NPUs / TPUs / DPUs (synthetic accelerators)
    if (cfg.enable_synthetic_accels) {
        for (int i = 0; i < cfg.npu_count; ++i) {
            DeviceInfo d;
            d.name       = npu_name(i);
            d.kind       = "npu";
            d.speed      = cfg.npu_speed;
            d.threads    = cfg.npu_threads;
            d.dram_cap   = cfg.npu_mem_capacity;
            d.dram_lat   = cfg.npu_mem_latency;
            d.dram_bw    = cfg.npu_mem_bandwidth;
            d.cache_cap  = cfg.npu_cache_capacity;
            d.cache_lat  = cfg.npu_cache_latency;
            d.cache_bw   = cfg.npu_cache_bandwidth;
            d.is_real    = false;
            devices.push_back(d);
        }

        for (int i = 0; i < cfg.tpu_count; ++i) {
            DeviceInfo d;
            d.name       = tpu_name(i);
            d.kind       = "tpu";
            d.speed      = cfg.tpu_speed;
            d.threads    = cfg.tpu_threads;
            d.dram_cap   = cfg.tpu_mem_capacity;
            d.dram_lat   = cfg.tpu_mem_latency;
            d.dram_bw    = cfg.tpu_mem_bandwidth;
            d.cache_cap  = cfg.tpu_cache_capacity;
            d.cache_lat  = cfg.tpu_cache_latency;
            d.cache_bw   = cfg.tpu_cache_bandwidth;
            d.is_real    = false;
            devices.push_back(d);
        }

        for (int i = 0; i < cfg.dpu_count; ++i) {
            DeviceInfo d;
            d.name       = dpu_name(i);
            d.kind       = "dpu";
            d.speed      = cfg.dpu_speed;
            d.threads    = cfg.dpu_threads;
            d.dram_cap   = cfg.dpu_mem_capacity;
            d.dram_lat   = cfg.dpu_mem_latency;
            d.dram_bw    = cfg.dpu_mem_bandwidth;
            d.cache_cap  = cfg.dpu_cache_capacity;
            d.cache_lat  = cfg.dpu_cache_latency;
            d.cache_bw   = cfg.dpu_cache_bandwidth;
            d.is_real    = false;
            devices.push_back(d);
        }
    }

    YAML::Node root;
    YAML::Node dev_nodes = YAML::Node(YAML::NodeType::Sequence);

    for (const auto& d : devices) {
        dev_nodes.push_back(device_node(d.name, d.kind, d.speed, d.threads,
                                        d.dram_cap, d.dram_lat, d.dram_bw,
                                        d.cache_cap, d.cache_lat, d.cache_bw));
    }
    root["devices"] = dev_nodes;

    YAML::Node topology = YAML::Node(YAML::NodeType::Sequence);
    std::map<std::string, DeviceInfo> dev_map;
    for (const auto& d : devices) dev_map[d.name] = d;

    for (const auto& a : dev_map) {
        for (const auto& b : dev_map) {
            if (a.first == b.first) continue;
            std::string kind_a = a.second.kind;
            std::string kind_b = b.second.kind;
            int speed, latency;
            std::string type;

            if (kind_a == "npu" || kind_b == "npu") {
                speed = cfg.npu_speed;
                latency = 3;
                type = "xgmi";
            } else if (kind_a == "gpu" && kind_b == "gpu") {
                speed = std::max(cfg.gpu_speed, 2000);
                latency = 1;
                type = "xgmi";
            } else if (kind_a == "tpu" || kind_b == "tpu") {
                speed = std::max(cfg.tpu_speed, 2000);
                latency = 1;
                type = "xgmi";
            } else if (kind_a == "dpu" || kind_b == "dpu") {
                speed = cfg.dpu_speed;
                latency = 8;
                type = "pci";
            } else {
                speed = cfg.cpu_speed;
                latency = 8;
                type = "pci";
            }
            topology.push_back(link_node(a.first, b.first, speed, latency, type));
        }
    }
    root["topology"] = topology;

    YAML::Node tasks = YAML::Node(YAML::NodeType::Sequence);
    auto add_task = [&](const std::string& name, int compute, int memory, int din, int dout,
                        const std::string& kind, int priority, int deadline, const std::string& device) {
        YAML::Node t;
        t["name"]           = name;
        t["compute"]        = compute;
        t["memory"]         = memory;
        t["data_in"]        = din;
        t["data_out"]       = dout;
        t["preferred_kind"] = kind;
        t["priority"]       = priority;
        t["deadline"]       = deadline;
        t["preemptible"]    = true;
        t["device"]         = device;
        tasks.push_back(t);
    };

    std::string dpu0 = (cfg.enable_synthetic_accels && cfg.dpu_count > 0) ? dpu_name(0) : "";
    std::string cpu0 = (cpus > 0) ? cpu_name(0) : "";
    std::string cpu1 = (cpus > 1) ? cpu_name(1) : cpu0;
    std::string gpu0 = (gpu_idx > 0) ? gpu_name(0) : "";
    std::string npu0 = (cfg.enable_synthetic_accels && cfg.npu_count > 0) ? npu_name(0) : "";
    std::string tpu0 = (cfg.enable_synthetic_accels && cfg.tpu_count > 0) ? tpu_name(0) : "";

    if (!dpu0.empty())
        add_task("ingest",      40,  512,  64,  64, "dpu", cfg.task_priority_base,                cfg.task_deadline_base,          dpu0);
    if (!cpu0.empty())
        add_task("decode",      80, 1024, 128, 128, "cpu", cfg.task_priority_base,                cfg.task_deadline_base + 2000,   cpu0);
    if (!cpu0.empty())
        add_task("preprocess", 120, 2048, 256, 256, "cpu", cfg.task_priority_base + 1,            cfg.task_deadline_base + 4000,   cpu0);
    if (!gpu0.empty())
        add_task("infer_a",    500, 4096, 512, 256, "gpu", 10,                                    cfg.task_deadline_base + 8000,   gpu0);
    if (!npu0.empty())
        add_task("infer_b",    420, 4096, 512, 256, "npu", 9,                                     cfg.task_deadline_base + 8000,   npu0);
    if (!tpu0.empty())
        add_task("fuse",        90, 1024, 128, 128, "tpu", 8,                                     cfg.task_deadline_base + 11000,  tpu0);
    if (!cpu1.empty())
        add_task("postprocess", 70, 1024, 128, 128, "cpu", 6,                                     cfg.task_deadline_base + 14000,  cpu1);
    if (!dpu0.empty())
        add_task("store",       30,  512,  64,  64, "dpu", 2,                                     cfg.task_deadline_base + 16000,  dpu0);

    root["tasks"] = tasks;

    YAML::Node deps = YAML::Node(YAML::NodeType::Sequence);
    auto add_dep = [&](const std::string& src, const std::string& dst, int bytes) {
        if (src.empty() || dst.empty()) return;
        YAML::Node d;
        d["src"]   = src;
        d["dst"]   = dst;
        d["bytes"] = bytes;
        deps.push_back(d);
    };

    add_dep("ingest",      "decode",      64);
    add_dep("decode",      "preprocess", 128);
    add_dep("preprocess",  "infer_a",    512);
    add_dep("preprocess",  "infer_b",    512);
    add_dep("infer_a",     "fuse",       128);
    add_dep("infer_b",     "fuse",       128);
    add_dep("fuse",        "postprocess", 64);
    add_dep("postprocess", "store",       32);
    root["dependencies"] = deps;

    std::ofstream fout(out);
    if (!fout) {
        std::cerr << "Cannot open output file: " << out << "\n";
        hwloc_topology_destroy(topo);
        return 1;
    }

    YAML::Emitter emitter;
    emitter << root;
    fout << emitter.c_str();

    if (!fout) {
        std::cerr << "Failed to write output file: " << out << "\n";
        hwloc_topology_destroy(topo);
        return 1;
    }

    std::cout << "Generated " << out << " with " << devices.size() << " devices ("
              << cpus << " CPUs, " << gpu_idx << " GPUs, "
              << (cfg.enable_synthetic_accels ? cfg.npu_count : 0) << " NPUs, "
              << (cfg.enable_synthetic_accels ? cfg.tpu_count : 0) << " TPUs, "
              << (cfg.enable_synthetic_accels ? cfg.dpu_count : 0) << " DPUs)\n";

    hwloc_topology_destroy(topo);
    return 0;
}
