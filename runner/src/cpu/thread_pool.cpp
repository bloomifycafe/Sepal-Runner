#include "sepal/cpu/thread_pool.hpp"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace sepal::cpu {

namespace {
thread_local bool in_task = false;
thread_local const void* confined_for = nullptr;  // the pool that confined this calling thread to its CPUs
constexpr int spin_rounds = 4000;

// An environment variable, "" when unset (MSVC deprecates getenv).
std::string env_string(const char* name) {
#if defined(_MSC_VER)
    char* env = nullptr;
    std::size_t length = 0;
    if (_dupenv_s(&env, &length, name) != 0 || env == nullptr) return {};
    std::string value = env;
    std::free(env);
    return value;
#else
    const char* env = std::getenv(name);  // NOLINT(concurrency-mt-unsafe): read at pool construction
    return env != nullptr ? std::string(env) : std::string();
#endif
}

std::string read_line(const std::string& path) {
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);
    return line;
}

#if defined(__linux__)
bool pin_to(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(static_cast<std::size_t>(cpu), &set);
    return pthread_setaffinity_np(pthread_self(), sizeof set, &set) == 0;
}

// The calling thread may run on any CPU of the pool, not on one: it also does the serial work between parallel steps
// (loading, merging), and a thread held on one CPU that something else uses (interrupts, a neighbour) runs at that CPU's
// pace. Measured: pinning the caller to the first CPU made loading 4B take 109 s instead of 35 s on a shared host.
bool confine_to(const std::vector<CpuSlot>& slots, unsigned count) {
    cpu_set_t set;
    CPU_ZERO(&set);
    for (unsigned i = 0; i < count && i < slots.size(); ++i) CPU_SET(static_cast<std::size_t>(slots[i].cpu), &set);
    return pthread_setaffinity_np(pthread_self(), sizeof set, &set) == 0;
}
#endif

// First threads of cores first, then by socket and core: participant i of a pinned pool takes slot i.
void order_slots(std::vector<CpuSlot>& slots) {
    std::stable_sort(slots.begin(), slots.end(), [](const CpuSlot& a, const CpuSlot& b) {
        if (a.first_thread != b.first_thread) return a.first_thread;
        if (a.package != b.package) return a.package < b.package;
        return a.core < b.core;
    });
}

std::string pin_mode() {
    const std::string v = env_string("SEPAL_CPU_PIN");
    if (v.empty()) return "auto";
    if (v == "0" || v == "off" || v == "no") return "off";
    if (v == "1" || v == "on" || v == "yes") return "on";
    return "auto";
}
} // namespace

std::vector<CpuSlot> cpu_slots() {
    std::vector<CpuSlot> out;
#if defined(__linux__)
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof allowed, &allowed) != 0) return out;
    std::set<std::pair<int, int>> cores_seen;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(static_cast<std::size_t>(cpu), &allowed)) continue;
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
        const std::string pkg = read_line(base + "physical_package_id"), core = read_line(base + "core_id");
        CpuSlot s;
        s.cpu = cpu;
        s.package = pkg.empty() ? 0 : std::atoi(pkg.c_str());
        s.core = core.empty() ? cpu : std::atoi(core.c_str());
        s.first_thread = cores_seen.insert({s.package, s.core}).second;
        out.push_back(s);
    }
    order_slots(out);
#endif
    return out;
}

std::vector<CpuSlot> cpu_slots(const std::string& list_arg) {
    std::string list = list_arg.empty() ? env_string("SEPAL_CPU_SET") : list_arg;
    std::vector<CpuSlot> all = cpu_slots();
    if (list.empty()) return all;
    std::set<int> wanted;
    std::size_t at = 0;
    const auto number = [&]() {
        if (at >= list.size() || list[at] < '0' || list[at] > '9') throw std::invalid_argument("CPU list \"" + list + "\": expected numbers like 0-11,24");
        int v = 0;
        while (at < list.size() && list[at] >= '0' && list[at] <= '9') v = v * 10 + (list[at++] - '0');
        return v;
    };
    while (at < list.size()) {
        if (list[at] == ',' || list[at] == ' ') {
            ++at;
            continue;
        }
        const int lo = number();
        int hi = lo;
        if (at < list.size() && list[at] == '-') {
            ++at;
            hi = number();
        }
        if (hi < lo || hi > 65535) throw std::invalid_argument("CPU list \"" + list + "\": bad range");
        for (int c = lo; c <= hi; ++c) wanted.insert(c);
    }
    std::vector<CpuSlot> out;
    std::set<std::pair<int, int>> cores_seen;
    for (const CpuSlot& s : all) {  // `all` lists first threads first, so a core keeps its first listed thread as the leader
        if (wanted.count(s.cpu) == 0) continue;
        CpuSlot t = s;
        t.first_thread = cores_seen.insert({s.package, s.core}).second;
        out.push_back(t);
    }
    order_slots(out);
    if (out.empty() && !all.empty()) throw std::invalid_argument("CPU list \"" + list + "\" names no CPU this process may use");
    return out;
}

unsigned physical_cores(const std::vector<CpuSlot>& slots) {
    return static_cast<unsigned>(std::count_if(slots.begin(), slots.end(), [](const CpuSlot& s) { return s.first_thread; }));
}

unsigned cgroup_cpu_limit() {
#if defined(__linux__)
    // cgroup v2: "max 100000" or "<quota> <period>"; v1: cfs_quota_us (-1 unlimited) and cfs_period_us.
    std::string line = read_line("/sys/fs/cgroup/cpu.max");
    double quota = -1, period = 0;
    if (!line.empty()) {
        std::istringstream in(line);
        std::string q;
        in >> q >> period;
        if (q != "max") quota = std::atof(q.c_str());
    } else {
        const std::string q = read_line("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), p = read_line("/sys/fs/cgroup/cpu/cpu.cfs_period_us");
        if (!q.empty() && !p.empty()) {
            quota = std::atof(q.c_str());
            period = std::atof(p.c_str());
        }
    }
    if (quota > 0 && period > 0) return static_cast<unsigned>(std::ceil(quota / period - 1e-9));
#endif
    return 0;
}

struct ThreadPool::Impl {
    unsigned participants = 1;
    std::vector<std::thread> threads;
    std::vector<CpuSlot> slots;  // participant i runs on slots[i] when pinned
    bool pinned = false;

    std::mutex mutex;
    std::condition_variable wake;
    std::uint64_t generation = 0; // guarded by `mutex` for writes; read atomically via `generation_hint`
    std::atomic<std::uint64_t> generation_hint{0};
    bool stop = false;

    Call call = nullptr;
    void* context = nullptr;
    std::size_t count = 0;
    std::atomic<std::size_t> next{0};
    std::atomic<unsigned> finished{0};

    std::mutex error_mutex;
    std::exception_ptr error;

    void drain(unsigned worker) {
        in_task = true;
        for (;;) {
            const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= count) break;
            try {
                call(context, i, worker);
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!error) error = std::current_exception();
                next.store(count, std::memory_order_relaxed);
            }
        }
        in_task = false;
    }

    void worker_main(unsigned worker) {
#if defined(__linux__)
        if (pinned) pin_to(slots[worker].cpu);
#endif
        std::uint64_t seen = 0;
        for (;;) {
            for (int spin = 0; spin < spin_rounds; ++spin) {
                if (generation_hint.load(std::memory_order_acquire) != seen) break;
                _mm_pause();
            }
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock, [&] { return stop || generation != seen; });
                if (stop) return;
                seen = generation;
            }
            drain(worker);
            finished.fetch_add(1, std::memory_order_release);
        }
    }
};

unsigned default_thread_count() {
    const unsigned logical = std::max(1u, std::thread::hardware_concurrency());
#if defined(__linux__)
    if (const std::vector<CpuSlot> slots = cpu_slots(); !slots.empty()) {
        const unsigned cores = physical_cores(slots);
        const unsigned quota = cgroup_cpu_limit();
        return std::max(1u, quota != 0 ? std::min(cores, quota) : cores);
    }
#endif
#if defined(_WIN32)
    DWORD bytes = 0;
    GetLogicalProcessorInformation(nullptr, &bytes);
    if (bytes == 0) return logical;
    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> info(bytes / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    if (!GetLogicalProcessorInformation(info.data(), &bytes)) return logical;
    unsigned cores = 0;
    for (const auto& i : info)
        if (i.Relationship == RelationProcessorCore) ++cores;
    return cores == 0 ? logical : std::min(cores, logical);
#elif defined(__linux__)
    // Count the allowed CPUs that are the first thread of their core (the lowest id in thread_siblings_list).
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof allowed, &allowed) != 0) return logical;
    unsigned cores = 0, seen = 0;
    const auto digits = [](const std::string& text, std::size_t& at) {
        std::size_t value = 0;
        while (at < text.size() && text[at] >= '0' && text[at] <= '9') value = value * 10 + static_cast<std::size_t>(text[at++] - '0');
        return value;
    };
    for (std::size_t cpu = 0; cpu < static_cast<std::size_t>(CPU_SETSIZE); ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) continue;
        ++seen;
        std::ifstream in("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list");
        std::string list;
        if (!std::getline(in, list)) {
            ++cores;  // no topology information: count the CPU itself
            continue;
        }
        bool leader = true;  // an allowed sibling with a lower id means this CPU is not the first thread of its core
        for (std::size_t p = 0; p < list.size();) {
            if (list[p] < '0' || list[p] > '9') {
                ++p;
                continue;
            }
            const std::size_t lo = digits(list, p);
            std::size_t hi = lo;
            if (p < list.size() && list[p] == '-') {
                ++p;
                hi = digits(list, p);
            }
            for (std::size_t c = lo; c <= hi && c < cpu; ++c) leader = leader && !CPU_ISSET(c, &allowed);
        }
        if (leader) ++cores;
    }
    return cores == 0 ? logical : std::min(cores, std::max(1u, seen));
#else
    return logical;
#endif
}

ThreadPool::ThreadPool(unsigned threads) : ThreadPool(threads, std::string()) {}

ThreadPool::ThreadPool(unsigned threads, const std::string& cpus) : impl_(std::make_unique<Impl>()) {
#if defined(__linux__)
    const bool explicit_set = !cpus.empty() || !env_string("SEPAL_CPU_SET").empty();
    impl_->slots = cpu_slots(cpus);
    if (threads == 0) {
        threads = default_thread_count();
        if (explicit_set && !impl_->slots.empty()) threads = std::max(1u, physical_cores(impl_->slots));
    }
    const std::string mode = pin_mode();
    if (mode != "off" && threads > 1 && !impl_->slots.empty()) {
        const unsigned cores = physical_cores(impl_->slots);
        const unsigned quota = cgroup_cpu_limit();
        const bool owns_cpus = explicit_set || quota == 0 || quota >= impl_->slots.size();
        impl_->pinned = mode == "on" || explicit_set ? threads <= impl_->slots.size() : (threads <= cores && owns_cpus);
    }
#else
    if (!cpus.empty()) throw std::invalid_argument("a CPU list is only supported on Linux");
    if (threads == 0) threads = default_thread_count();
#endif
    impl_->participants = threads;
    for (unsigned w = 1; w < threads; ++w) impl_->threads.emplace_back([this, w] { impl_->worker_main(w); });
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->stop = true;
    }
    impl_->wake.notify_all();
    for (std::thread& t : impl_->threads) t.join();
}

unsigned ThreadPool::size() const { return impl_->participants; }

bool ThreadPool::pinned() const { return impl_->pinned; }

std::string ThreadPool::layout() const {
    const std::vector<CpuSlot> slots = impl_->slots.empty() ? cpu_slots() : impl_->slots;
    std::ostringstream out;
    out << impl_->participants << (impl_->participants == 1 ? " thread" : " threads");
    if (!slots.empty()) {
        std::set<std::pair<int, int>> cores;
        std::set<int> packages;
        for (const CpuSlot& s : slots) {
            cores.insert({s.package, s.core});
            packages.insert(s.package);
        }
        out << "; allowed: " << slots.size() << " logical CPUs on " << cores.size() << " physical cores, " << packages.size()
            << (packages.size() == 1 ? " socket" : " sockets");
    }
    if (const unsigned quota = cgroup_cpu_limit(); quota != 0) out << "; cgroup quota " << quota << " CPUs";
    if (impl_->pinned) {
        std::set<std::pair<int, int>> cores;
        std::set<int> packages;
        out << "; one thread per core on CPUs";
        for (unsigned i = 0; i < impl_->participants; ++i) {
            out << (i == 0 ? " " : ",") << slots[i].cpu;
            cores.insert({slots[i].package, slots[i].core});
            packages.insert(slots[i].package);
        }
        out << " (" << cores.size() << " cores, " << packages.size() << (packages.size() == 1 ? " socket" : " sockets")
            << "; workers pinned, the caller kept on these CPUs)";
    } else {
        out << "; not pinned (SEPAL_CPU_PIN=" << pin_mode() << ")";
    }
    return out.str();
}

void ThreadPool::run(std::size_t n, Call call, void* context) {
    if (n == 0) return;
    if (in_task || impl_->participants == 1 || n == 1) {
        const bool was_in_task = in_task;
        in_task = true;
        try {
            for (std::size_t i = 0; i < n; ++i) call(context, i, 0);
        } catch (...) {
            in_task = was_in_task;
            throw;
        }
        in_task = was_in_task;
        return;
    }
    Impl& s = *impl_;
#if defined(__linux__)
    if (s.pinned && confined_for != impl_.get()) {
        confine_to(s.slots, s.participants);
        confined_for = impl_.get();
    }
#endif
    s.call = call;
    s.context = context;
    s.count = n;
    s.error = nullptr;
    s.next.store(0, std::memory_order_relaxed);
    s.finished.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        ++s.generation;
        s.generation_hint.store(s.generation, std::memory_order_release);
    }
    s.wake.notify_all();
    s.drain(0);
    const unsigned workers = s.participants - 1;
    while (s.finished.load(std::memory_order_acquire) != workers) _mm_pause();
    if (s.error) {
        std::exception_ptr e = s.error;
        s.error = nullptr;
        std::rethrow_exception(e);
    }
}

} // namespace sepal::cpu
