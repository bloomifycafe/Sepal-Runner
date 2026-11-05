// A persistent thread pool for the CPU backend.
//
// Determinism rule (RUNNER.md 14): a task writes only outputs of its own and computes each output with a summation
// order that does not depend on which thread runs it or on how many threads there are. The pool only distributes
// whole tasks (dynamically, by an atomic counter), so no thread count can change the order of any sum.
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace sepal::cpu {

/// The default number of worker threads: the physical cores this process may use (hyper-threads share the FMA units, and
/// a spinning pool oversubscribed with them is slower than one thread per core), capped by a cgroup CPU quota (a container
/// limited to 10 CPUs of a 40-core host gets 10). Falls back to the hardware concurrency.
unsigned default_thread_count();

/// The CPUs this process may run on (Linux: affinity mask and sysfs topology), ordered for pinning: the first thread of
/// every physical core, by socket then core, then the remaining hyper-threads. Empty where the topology is unknown.
struct CpuSlot {
    int cpu = 0, package = 0, core = 0;
    bool first_thread = true;  // the lowest allowed CPU of its core
};
std::vector<CpuSlot> cpu_slots();

/// `cpu_slots()` restricted to the CPUs of `list` ("0-11", "0-9,20-29", or "" for all allowed ones; SEPAL_CPU_SET is used when
/// `list` is empty). Throws std::invalid_argument for a malformed list or one that names no allowed CPU.
std::vector<CpuSlot> cpu_slots(const std::string& list);

/// The physical cores among `slots` (first threads).
unsigned physical_cores(const std::vector<CpuSlot>& slots);

/// CPUs granted by a cgroup quota (cpu.max or cfs_quota_us / cfs_period_us), rounded up; 0 when unlimited or unknown.
unsigned cgroup_cpu_limit();

class ThreadPool {
public:
    /// `threads` is the number of participants including the calling thread; 0 means default_thread_count().
    /// Pinning (Linux; SEPAL_CPU_PIN=auto|on|off, default auto): participant i runs on the i-th CPU of `cpu_slots()`, so a
    /// pool of at most one thread per physical core never puts two threads on one core and fills one socket before the
    /// next. Auto pins when the pool fits in the physical cores and no cgroup quota is smaller than the allowed CPU set
    /// (inside such a container the CPUs are shared, and the scheduler is left to place the threads). Participants 1..n-1
    /// are pinned to slots 1..n-1; the calling thread is confined to the pool's CPUs by its first parallel_for (it floats
    /// among them and stays off the CPUs outside the pool).
    explicit ThreadPool(unsigned threads = 0);

    /// The same on the CPUs of `cpus` (cpu_slots(cpus)): with a list, threads 0 means one per physical core of the list,
    /// and the pool pins unless SEPAL_CPU_PIN=off. Use it to keep the runner off the cores of another engine on the machine.
    ThreadPool(unsigned threads, const std::string& cpus);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    unsigned size() const;

    /// True when the workers are pinned to their own CPUs (one per physical core while they last) and the caller confined to the pool's.
    bool pinned() const;

    /// One line for logs: threads, cores and sockets used, pinning, and the cgroup quota when there is one.
    std::string layout() const;

    /// Runs `f(index, worker)` for every index in [0, n), each exactly once. `worker` is in [0, size()) and is
    /// stable for the duration of one call (use it to index per-thread scratch). Blocks until all are done; the
    /// first exception thrown by a task is rethrown here. A nested call from inside a task runs serially.
    template <class F>
    void parallel_for(std::size_t n, F&& f) {
        run(n, &trampoline<std::remove_reference_t<F>>, static_cast<void*>(&f));
    }

private:
    template <class F>
    static void trampoline(void* context, std::size_t index, unsigned worker) {
        (*static_cast<F*>(context))(index, worker);
    }
    using Call = void (*)(void*, std::size_t, unsigned);
    void run(std::size_t n, Call call, void* context);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sepal::cpu
