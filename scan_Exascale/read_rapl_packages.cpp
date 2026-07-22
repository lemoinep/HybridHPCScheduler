#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

// MSR constants for Intel RAPL (package)
static const uint32_t MSR_RAPL_POWER_UNIT      = 0x606;
static const uint32_t MSR_PKG_ENERGY_STATUS    = 0x611;

// Read a 64-bit MSR value on a given core
bool read_msr(int core, uint32_t msr, uint64_t &value) {
    std::string path = "/dev/cpu/" + std::to_string(core) + "/msr";
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        std::cerr << "Failed to open " << path
                  << " (check /dev/cpu/*/msr, msr module, and permissions)." << std::endl;
        return false;
    }
    ssize_t n = pread(fd, &value, sizeof(value), msr);
    close(fd);
    if (n != sizeof(value)) {
        std::cerr << "Failed to read MSR 0x" << std::hex << msr
                  << " on core " << std::dec << core << std::endl;
        return false;
    }
    return true;
}

// Detect packages via sysfs: /sys/devices/system/cpu/cpuX/topology/physical_package_id
std::map<int, std::vector<int>> detect_packages() {
    std::map<int, std::vector<int>> package_to_cores;

    const std::string base = "/sys/devices/system/cpu";
    if (!fs::exists(base)) {
        std::cerr << "Sysfs path " << base << " does not exist." << std::endl;
        return package_to_cores;
    }

    for (const auto &entry : fs::directory_iterator(base)) {
        if (!entry.is_directory()) continue;
        std::string name = entry.path().filename().string();
        if (name.rfind("cpu", 0) != 0) continue; // must start with "cpu"
        // extract core index
        int core = -1;
        try {
            core = std::stoi(name.substr(3));
        } catch (...) {
            continue;
        }

        std::string topo_path = entry.path().string() + "/topology/physical_package_id";
        FILE *f = fopen(topo_path.c_str(), "r");
        if (!f) {
            continue;
        }
        int pkg_id = 0;
        if (fscanf(f, "%d", &pkg_id) != 1) {
            fclose(f);
            continue;
        }
        fclose(f);

        package_to_cores[pkg_id].push_back(core);
    }

    return package_to_cores;
}

int main(int argc, char **argv) {
    // Optional: allow interval override via argument (seconds)
    double interval_seconds = 0.1; // default 100 ms
    if (argc >= 2) {
        try {
            interval_seconds = std::stod(argv[1]);
        } catch (...) {
            std::cerr << "Invalid interval argument, using default 0.1 s." << std::endl;
        }
        if (interval_seconds <= 0.0) {
            interval_seconds = 0.1;
        }
    }

    // 1. Detect CPU packages and representative cores
    auto packages = detect_packages();
    if (packages.empty()) {
        std::cerr << "No CPU packages detected (check sysfs / permissions)." << std::endl;
        return 1;
    }

    // 2. Read MSR_RAPL_POWER_UNIT on a reference core to get energy unit
    int ref_core = packages.begin()->second.front();
    uint64_t power_unit_raw = 0;
    if (!read_msr(ref_core, MSR_RAPL_POWER_UNIT, power_unit_raw)) {
        std::cerr << "Failed to read MSR_RAPL_POWER_UNIT (0x606)." << std::endl;
        return 1;
    }

    // energy_units_raw is in bits 8-12 (5 bits), see Intel RAPL docs
    uint32_t energy_units_raw = (power_unit_raw >> 8) & 0x1F;
    double energy_unit_joules = 1.0 / (1u << energy_units_raw); // J per tick

    // 3. Take initial energy snapshot for each package
    std::map<int, uint64_t> energy_start;
    for (const auto &kv : packages) {
        int pkg_id = kv.first;
        const auto &cores = kv.second;
        if (cores.empty()) continue;
        int core = cores.front(); // representative core in this package

        uint64_t val = 0;
        if (!read_msr(core, MSR_PKG_ENERGY_STATUS, val)) {
            std::cerr << "Failed to read MSR_PKG_ENERGY_STATUS (0x611) for package "
                      << pkg_id << " on core " << core << std::endl;
            return 1;
        }
        energy_start[pkg_id] = val;
    }

    // 4. Wait interval_seconds
    auto sleep_ms = static_cast<int>(interval_seconds * 1000.0);
    if (sleep_ms <= 0) {
        sleep_ms = 100;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));

    // 5. Take final snapshot and compute power
    std::map<int, double> power_watts;
    for (const auto &kv : packages) {
        int pkg_id = kv.first;
        const auto &cores = kv.second;
        if (cores.empty()) continue;
        int core = cores.front();

        uint64_t val_end = 0;
        if (!read_msr(core, MSR_PKG_ENERGY_STATUS, val_end)) {
            std::cerr << "Failed to read MSR_PKG_ENERGY_STATUS (end) for package "
                      << pkg_id << " on core " << core << std::endl;
            return 1;
        }

        uint64_t val_start = energy_start[pkg_id];

        // Simple wrap-around handling: if val_end < val_start, assume a single wrap
        uint64_t delta_raw;
        if (val_end >= val_start) {
            delta_raw = val_end - val_start;
        } else {
            delta_raw = (UINT64_MAX - val_start) + val_end + 1;
        }

        double delta_joules = static_cast<double>(delta_raw) * energy_unit_joules;
        double power = delta_joules / interval_seconds; // W = J / s
        power_watts[pkg_id] = power;
    }

    // 6. Output CSV to stdout
    std::cout << "package_id,power_watts\n";
    for (const auto &kv : power_watts) {
        std::cout << kv.first << "," << kv.second << "\n";
    }

    return 0;
}

