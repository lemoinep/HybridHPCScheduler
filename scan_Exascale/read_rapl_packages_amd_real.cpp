#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
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
                while (!val.empty() && (val.back() == ' ' || val.back() == '\t' ||
                                         val.back() == '\n' || val.back() == '\r')) {
                    val.pop_back();
                }
                return val;
            }
        }
    }
    return "";
}

int main(int argc, char **argv) {
    // Arguments:
    //   argv[1] = interval_seconds (optional, default 0.1)
    //   argv[2] = num_packages (optional, default 2)
    double interval_seconds = 0.1;
    int num_packages = 2;

    if (argc >= 2) {
        try {
            double val = std::stod(argv[1]);
            if (val > 0.0) {
                interval_seconds = val;
            }
        } catch (...) {
            std::cerr << "Warning: invalid interval argument, using default 0.1 s.\n";
        }
    }
    if (argc >= 3) {
        try {
            int n = std::stoi(argv[2]);
            if (n > 0) {
                num_packages = n;
            }
        } catch (...) {
            std::cerr << "Warning: invalid packages argument, using default 2.\n";
        }
    }

    // Check that we are on AMD
    std::string vendor = get_cpu_vendor();
    if (!vendor.empty() && vendor != "AuthenticAMD") {
        std::cerr << "Error: CPU vendor is not AuthenticAMD (detected: " << vendor << ").\n";
        std::cerr << "This binary is intended for AMD CPUs.\n";
        return 1;
    }

    // Build perf command:
    // perf stat -a -e power/energy-pkg/ sleep <interval> 2>&1
    char cmd[256];
    std::snprintf(
        cmd,
        sizeof(cmd),
        "perf stat -a -e power/energy-pkg/ sleep %.3f 2>&1",
        interval_seconds
    );

    FILE *pipe = popen(cmd, "r");
    if (!pipe) {
        std::cerr << "Failed to run perf stat command.\n";
        return 1;
    }

    std::string output;
    char buf[4096];
    while (fgets(buf, sizeof(buf), pipe)) {
        output += buf;
    }
    int rc = pclose(pipe);
    if (rc != 0) {
        std::cerr << "Warning: perf stat returned non-zero exit code (" << rc << ").\n";
        // On continue quand même en essayant de parser la sortie.
    }

    // Parse Joules from perf output.
    // We look for the token "Joules" and extract the number before it.
    double joules = 0.0;
    std::size_t pos = output.find("Joules");
    if (pos == std::string::npos) {
        std::cerr << "Error: no 'Joules' token found in perf output.\n";
        // Option de repli: on pourrait mettre une valeur par défaut, mais on préfère signaler l'erreur.
        return 1;
    } else {
        // Seek backwards from pos to find the start of the number
        std::size_t start = pos;
        // Skip spaces/tabs immediately before "Joules"
        while (start > 0 && (output[start - 1] == ' ' || output[start - 1] == '\t')) {
            --start;
        }
        // Now go further back while characters are part of the numeric token (digits or '.')
        std::size_t end = start;
        while (end > 0 && (std::isdigit(static_cast<unsigned char>(output[end - 1])) ||
                           output[end - 1] == '.')) {
            --end;
        }
        if (end == start) {
            std::cerr << "Error: failed to locate numeric value before 'Joules'.\n";
            return 1;
        }
        std::string num_str = output.substr(end, start - end);
        try {
            joules = std::stod(num_str);
        } catch (...) {
            std::cerr << "Error: failed to parse Joules value '" << num_str << "'.\n";
            return 1;
        }
    }

    // Convert Joules to total power (Watts)
    double total_power = joules / interval_seconds;

    // For now, split evenly across num_packages.
    double per_pkg_power = total_power / static_cast<double>(num_packages);

    // Output CSV to stdout: package_id,power_watts
    std::cout << "package_id,power_watts\n";
    for (int pkg = 0; pkg < num_packages; ++pkg) {
        std::cout << pkg << "," << per_pkg_power << "\n";
    }

    return 0;
}

