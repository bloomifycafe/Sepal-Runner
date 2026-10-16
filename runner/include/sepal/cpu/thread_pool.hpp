// A persistent thread pool for the CPU backend.
//
// Determinism rule (RUNNER.md 14): a task writes only outputs of its own and computes each output with a summation
// order that does not depend on which thread runs it or on how many threads there are. The pool only distributes
// whole tasks (dynamically, by an atomic counter), so no thread count can change the order of any sum.
#pragma once

#include <cstddef>
#include <memory>
#include <utility>

namespace sepal::cpu {

/// The default number of worker threads: the physical cores this process may use (hyper-threads share the FMA units, and
/// a spinning pool oversubscribed with them is slower than one thread per core). Falls back to the hardware concurrency.
unsigned default_thread_count();

class ThreadPool {
public:
    /// `threads` is the number of participants including the calling thread; 0 means default_thread_count().
    explicit ThreadPool(unsigned threads = 0);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    unsigned size() const;

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
