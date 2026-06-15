
// Author(s): Dr. Patrick Lemoine

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <optional>
#include <algorithm>
#include <cctype>
#include <iomanip>
#include <cstdlib>

namespace fs = std::filesystem;

static std::string read_file(const fs::path& p) {
    std::ifstream f(p);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    while (!s.empty() && (s.back()=='\n' || s.back()=='\r' || s.back()==' ' || s.back()=='\t'))
        s.pop_back();
    return s;
}

static std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
        [](unsigned char c){ return std::tolower(c); });
    return s;
}

static bool icontains(const std::string& hay, const std::string& needle) {
    return lower(hay).find(lower(needle)) != std::string::npos;
}

static std::string trim(std::string s) {
    auto issp = [](unsigned char c){ return std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [&](unsigned char c){ return !issp(c); }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [&](unsigned char c){ return !issp(c); }).base(), s.end());
    return s;
}

static std::string symlink_name(const fs::path& p) {
    try { if (fs::exists(p)) return fs::read_symlink(p).filename().string(); } catch (...) {}
    return {};
}

static std::string shell(const std::string& cmd) {
    FILE* fp = popen(cmd.c_str(), "r");
    if (!fp) return {};
    char buf[4096];
    std::string out;
    while (fgets(buf, sizeof(buf), fp)) out += buf;
    pclose(fp);
    return trim(out);
}

static std::vector<std::string> lines(const std::string& s) {
    std::istringstream iss(s);
    std::vector<std::string> out;
    for (std::string x; std::getline(iss, x);) out.push_back(x);
    return out;
}

static std::vector<std::string> split_ws(const std::string& s) {
    std::istringstream iss(s);
    std::vector<std::string> out;
    for (std::string w; iss >> w;) out.push_back(w);
    return out;
}

static std::string hexnorm(const std::string& s) {
    std::string x = trim(s);
    if (x.rfind("0x", 0) == 0 || x.rfind("0X", 0) == 0) x = x.substr(2);
    return lower(x);
}

struct PciDb {
    std::map<std::string, std::string> vendors;
    std::map<std::string, std::map<std::string, std::string>> devices;
    std::map<std::string, std::string> classes;
};

struct UsbDb {
    std::map<std::string, std::string> vendors;
    std::map<std::string, std::map<std::string, std::string>> products;
};

static std::optional<fs::path> first_existing(const std::vector<fs::path>& cands) {
    for (const auto& p : cands) if (fs::exists(p)) return p;
    return std::nullopt;
}

static PciDb load_pci_ids(const fs::path& file) {
    PciDb db;
    std::ifstream f(file);
    if (!f) return db;
    std::string line, cur_vendor, cur_class;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (line[0] == 'C' && line.size() > 2 && line[1] == ' ') {
            auto p = split_ws(line);
            if (p.size() >= 3) {
                cur_class = lower(p[1]);
                db.classes[cur_class] = line.substr(line.find(p[2]));
            }
            continue;
        }
        if (line[0] != '\t') {
            auto p = split_ws(line);
            if (p.size() >= 2) {
                cur_vendor = lower(p[0]);
                db.vendors[cur_vendor] = line.substr(line.find(p[1]));
            }
            continue;
        }
        if (line.size() >= 2 && line[1] != '\t') {
            auto p = split_ws(line);
            if (p.size() >= 2) db.devices[cur_vendor][lower(p[0])] = line.substr(line.find(p[1]));
        }
    }
    return db;
}

static UsbDb load_usb_ids(const fs::path& file) {
    UsbDb db;
    std::ifstream f(file);
    if (!f) return db;
    std::string line, cur_vendor;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (line[0] != '\t') {
            auto p = split_ws(line);
            if (p.size() >= 2) {
                cur_vendor = lower(p[0]);
                db.vendors[cur_vendor] = line.substr(line.find(p[1]));
            }
            continue;
        }
        if (line.size() >= 2 && line[1] != '\t') {
            auto p = split_ws(line);
            if (p.size() >= 2) db.products[cur_vendor][lower(p[0])] = line.substr(line.find(p[1]));
        }
    }
    return db;
}

struct DeviceInfo {
    std::string bus, path, addr;
    std::string vendor_id, device_id, subvendor_id, subdevice_id, class_id;
    std::string driver, name, modalias;
    std::string vendor_name, device_name;
    std::string kind;
    double confidence = 0.0;
};

static DeviceInfo load_pci(const fs::path& p) {
    DeviceInfo d;
    d.bus = "pci";
    d.path = p.string();
    d.addr = p.filename().string();
    d.vendor_id = hexnorm(read_file(p / "vendor"));
    d.device_id = hexnorm(read_file(p / "device"));
    d.subvendor_id = hexnorm(read_file(p / "subsystem_vendor"));
    d.subdevice_id = hexnorm(read_file(p / "subsystem_device"));
    d.class_id = hexnorm(read_file(p / "class"));
    d.driver = symlink_name(p / "driver");
    d.name = read_file(p / "product");
    d.modalias = read_file(p / "modalias");
    return d;
}

static DeviceInfo load_usb(const fs::path& p) {
    DeviceInfo d;
    d.bus = "usb";
    d.path = p.string();
    d.addr = p.filename().string();
    d.vendor_id = hexnorm(read_file(p / "idVendor"));
    d.device_id = hexnorm(read_file(p / "idProduct"));
    d.driver = symlink_name(p / "driver");
    d.name = read_file(p / "product");
    d.modalias = read_file(p / "uevent");
    return d;
}

static void enrich(DeviceInfo& d, const PciDb& pci, const UsbDb& usb) {
    if (d.bus == "pci") {
        if (auto it = pci.vendors.find(d.vendor_id); it != pci.vendors.end()) d.vendor_name = it->second;
        if (auto iv = pci.devices.find(d.vendor_id); iv != pci.devices.end()) {
            if (auto id = iv->second.find(d.device_id); id != iv->second.end()) d.device_name = id->second;
        }
    } else if (d.bus == "usb") {
        if (auto it = usb.vendors.find(d.vendor_id); it != usb.vendors.end()) d.vendor_name = it->second;
        if (auto iv = usb.products.find(d.vendor_id); iv != usb.products.end()) {
            if (auto id = iv->second.find(d.device_id); id != iv->second.end()) d.device_name = id->second;
        }
    }
}

static std::string classify(DeviceInfo& d) {
    std::string blob = d.vendor_id + " " + d.device_id + " " + d.class_id + " " + d.driver + " " +
                       d.name + " " + d.vendor_name + " " + d.device_name + " " + d.modalias;
    d.confidence = 0.0;

    auto set = [&](const std::string& k, double c){ d.kind = k; d.confidence = c; };

    if (d.bus == "usb") {
        if (icontains(blob, "coral") || icontains(blob, "edgetpu") || icontains(blob, "google tpu")) {
            set("TPU", 0.98);
            return d.kind;
        }
    }

    if (d.bus == "pci") {
        if (icontains(blob, "amdxdna") || icontains(blob, "ryzen ai")) { set("NPU", 0.97); return d.kind; }
        if (icontains(blob, "intel_vpu") || icontains(blob, "intel npu")) { set("NPU", 0.97); return d.kind; }
        if (icontains(blob, "bluefield") || icontains(blob, "mellanox")) { set("DPU", 0.95); return d.kind; }
        if (icontains(blob, "mlx") && icontains(blob, "network")) { set("DPU", 0.85); return d.kind; }
        if (d.class_id.rfind("0x02", 0) == 0) { set("DPU / network accelerator probable", 0.60); return d.kind; }
        if (d.class_id.rfind("0x12", 0) == 0) { set("accelerator", 0.50); return d.kind; }
    }

    d.kind = "unknown";
    d.confidence = 0.0;
    return d.kind;
}

static std::string esc(const std::string& s) {
    std::ostringstream o;
    for (unsigned char c : s) {
        switch (c) {
            case '\\': o << "\\\\"; break;
            case '"': o << "\\\""; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default: o << c;
        }
    }
    return o.str();
}

static std::string detect_hostname() {
    return trim(shell("hostname"));
}

static std::string detect_kernel() {
    return trim(shell("uname -r"));
}

static std::string detect_distro() {
    std::ifstream f("/etc/os-release");
    if (!f) return {};
    std::string line, name, ver;
    while (std::getline(f, line)) {
        if (line.rfind("PRETTY_NAME=", 0) == 0) {
            name = line.substr(13);
            if (!name.empty() && name.front() == '"') name.erase(0,1);
            if (!name.empty() && name.back() == '"') name.pop_back();
            return name;
        }
    }
    return {};
}

static void collect_dmesg(std::vector<std::string>& out) {
    auto s = shell("dmesg 2>/dev/null | grep -Ei 'amdxdna|intel_vpu|npu|tpu|dpu|vpu' | tail -n 20");
    for (auto& l : lines(s)) if (!trim(l).empty()) out.push_back(l);
}

int main(int argc, char** argv) {
    bool json = false, verbose = false;
    std::string csv_out;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--json") json = true;
        else if (a == "--verbose") verbose = true;
        else if (a == "--csv" && i + 1 < argc) csv_out = argv[++i];
    }

    auto pci_path = first_existing({"/usr/share/hwdata/pci.ids", "/usr/share/misc/pci.ids", "/usr/share/pci.ids"});
    auto usb_path = first_existing({"/usr/share/hwdata/usb.ids", "/var/lib/usbutils/usb.ids", "/usr/share/misc/usb.ids"});
    PciDb pci = pci_path ? load_pci_ids(*pci_path) : PciDb{};
    UsbDb usb = usb_path ? load_usb_ids(*usb_path) : UsbDb{};

    std::vector<DeviceInfo> devs;
    if (fs::exists("/sys/bus/pci/devices")) {
        for (const auto& e : fs::directory_iterator("/sys/bus/pci/devices")) {
            if (!e.is_directory()) continue;
            auto d = load_pci(e.path());
            enrich(d, pci, usb);
            if (classify(d) != "unknown") devs.push_back(d);
        }
    }
    if (fs::exists("/sys/bus/usb/devices")) {
        for (const auto& e : fs::directory_iterator("/sys/bus/usb/devices")) {
            if (!e.is_directory()) continue;
            auto d = load_usb(e.path());
            enrich(d, pci, usb);
            if (classify(d) != "unknown") devs.push_back(d);
        }
    }

    std::vector<std::string> dm;
    collect_dmesg(dm);

    if (!csv_out.empty()) {
        std::ofstream f(csv_out);
        f << "type,confidence,bus,addr,path,vendor_id,vendor_name,device_id,device_name,subvendor_id,subdevice_id,class_id,driver,name,modalias\n";
        for (auto d : devs) {
            auto t = classify(d);
            f << '"' << t << "\","
              << std::fixed << std::setprecision(2) << d.confidence << ','
              << '"' << d.bus << "\","
              << '"' << d.addr << "\","
              << '"' << d.path << "\","
              << '"' << d.vendor_id << "\","
              << '"' << d.vendor_name << "\","
              << '"' << d.device_id << "\","
              << '"' << d.device_name << "\","
              << '"' << d.subvendor_id << "\","
              << '"' << d.subdevice_id << "\","
              << '"' << d.class_id << "\","
              << '"' << d.driver << "\","
              << '"' << d.name << "\","
              << '"' << d.modalias << "\"\n";
        }
    }

    if (json) {
        std::cout << "{";
        std::cout << "\"host\":\"" << esc(detect_hostname()) << "\",";
        std::cout << "\"kernel\":\"" << esc(detect_kernel()) << "\",";
        std::cout << "\"distro\":\"" << esc(detect_distro()) << "\",";
        std::cout << "\"pci_ids\":\"" << esc(pci_path ? pci_path->string() : "") << "\",";
        std::cout << "\"usb_ids\":\"" << esc(usb_path ? usb_path->string() : "") << "\",";
        std::cout << "\"devices\":[";
        for (size_t i = 0; i < devs.size(); ++i) {
            auto d = devs[i];
            auto t = classify(d);
            if (i) std::cout << ",";
            std::cout << "{"
                      << "\"type\":\"" << esc(t) << "\","
                      << "\"confidence\":" << std::fixed << std::setprecision(2) << d.confidence << ","
                      << "\"bus\":\"" << esc(d.bus) << "\","
                      << "\"addr\":\"" << esc(d.addr) << "\","
                      << "\"path\":\"" << esc(d.path) << "\","
                      << "\"vendor_id\":\"" << esc(d.vendor_id) << "\","
                      << "\"vendor_name\":\"" << esc(d.vendor_name) << "\","
                      << "\"device_id\":\"" << esc(d.device_id) << "\","
                      << "\"device_name\":\"" << esc(d.device_name) << "\","
                      << "\"subvendor_id\":\"" << esc(d.subvendor_id) << "\","
                      << "\"subdevice_id\":\"" << esc(d.subdevice_id) << "\","
                      << "\"class_id\":\"" << esc(d.class_id) << "\","
                      << "\"driver\":\"" << esc(d.driver) << "\","
                      << "\"name\":\"" << esc(d.name) << "\","
                      << "\"modalias\":\"" << esc(d.modalias) << "\""
                      << "}";
        }
        std::cout << "],";
        std::cout << "\"dmesg\":[";
        for (size_t i = 0; i < dm.size(); ++i) {
            if (i) std::cout << ",";
            std::cout << "\"" << esc(dm[i]) << "\"";
        }
        std::cout << "]";
        std::cout << "}\n";
        return 0;
    }

    std::cout << "Host: " << detect_hostname() << "\n";
    std::cout << "Kernel: " << detect_kernel() << "\n";
    std::cout << "Distro: " << detect_distro() << "\n";
    std::cout << "PCI IDs: " << (pci_path ? pci_path->string() : "not found") << "\n";
    std::cout << "USB IDs: " << (usb_path ? usb_path->string() : "not found") << "\n";
    std::cout << "\nDetected accelerators:\n";

    if (devs.empty()) {
        std::cout << "  none\n";
    } else {
        for (auto& d : devs) {
            auto t = classify(d);
            if (verbose) {
                std::cout << "Type: " << t << "\n";
                std::cout << "Confidence: " << std::fixed << std::setprecision(2) << d.confidence << "\n";
                std::cout << "Bus: " << d.bus << "  Addr: " << d.addr << "\n";
                std::cout << "Vendor: " << d.vendor_id << "  " << d.vendor_name << "\n";
                std::cout << "Device: " << d.device_id << "  " << d.device_name << "\n";
                std::cout << "Subsystem: " << d.subvendor_id << ":" << d.subdevice_id << "\n";
                std::cout << "Class: " << d.class_id << "\n";
                std::cout << "Driver: " << d.driver << "\n";
                std::cout << "Sysfs path: " << d.path << "\n";
                if (!d.name.empty()) std::cout << "Name: " << d.name << "\n";
                if (!d.modalias.empty()) std::cout << "Modalias: " << d.modalias << "\n";
                std::cout << "-----\n";
            } else {
                std::cout << "  " << t << " " << d.vendor_id << ":" << d.device_id
                          << " conf=" << std::fixed << std::setprecision(2) << d.confidence;
                if (!d.vendor_name.empty() || !d.device_name.empty())
                    std::cout << " (" << d.vendor_name << " " << d.device_name << ")";
                std::cout << "\n";
            }
        }
    }

    if (!dm.empty()) {
        std::cout << "\nRelevant dmesg lines:\n";
        for (const auto& l : dm) std::cout << "  " << l << "\n";
    }

    return 0;
}