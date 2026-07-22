#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>

// Simple helper to read vendor_id from /proc/cpuinfo
std::string get_cpu_vendor() {
    std::ifstream f("/proc/cpuinfo");
    if (!f.is_open()) {
        return "";
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("vendor_id", 0) == 0) {
            // line is like: vendor_id    : AuthenticAMD
            auto pos = line.find(':');
            if (pos != std::string::npos) {
                std::string val = line.substr(pos + 1);
                // trim spaces
                while (!val.empty() && (val.front() == ' ' || val.front() == '\t')) {
                    val.erase(val.begin());
                }
                while (!val.empty() && (val.back() == ' ' || val.back() == '\t' || val.back() == '\n' || val.back() == '\r')) {
                    val.pop_back();
                }
                return val;
            }
        }
    }
    return "";
}

int main(int argc, char **argv) {
    // Check vendor_id
    std::string vendor = get_cpu_vendor();
    if (vendor.empty()) {
        std::cerr << "Warning: unable to read /proc/cpuinfo vendor_id, assuming AMD stub.\n";
    } else if (vendor != "AuthenticAMD") {
        std::cerr << "Error: CPU vendor is not AuthenticAMD (detected: " << vendor << ").\n";
        std::cerr << "This binary is intended as an AMD-specific stub.\n";
        return 1;
    }

    // For now, this is a stub: we simulate package powers.
    // You can adjust these values or number of packages as needed.
    // Later, you will replace this with a real implementation using perf_event_open
    // and the energy-pkg perf events available on AMD Zen CPUs.

    // Number of packages: for a stub, we can guess 2, or accept an optional argument.
    int num_packages = 2;
    if (argc >= 2) {
        try {
            int n = std::stoi(argv[1]);
            if (n > 0) {
                num_packages = n;
            }
        } catch (...) {
            // ignore, keep default
        }
    }

    // Simulated power range (in Watts) for AMD packages.
    double power_min = 80.0;
    double power_max = 220.0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(power_min, power_max);

    std::cout << "package_id,power_watts\n";
    for (int pkg = 0; pkg < num_packages; ++pkg) {
        double p = dist(gen);
        std::cout << pkg << "," << p << "\n";
    }

    return 0;
}

