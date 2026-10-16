#include "sepal/cpu/thread_pool.hpp"

#include <intrin.h>

#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace sepal::cpu {

namespace {
thread_local bool in_task = false;
constexpr int spin_rounds = 4000;
} // namespace

struct ThreadPool::Impl {
    unsigned participants = 1;
    std::vector<std::thread> threads;

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

ThreadPool::ThreadPool(unsigned threads) : impl_(std::make_unique<Impl>()) {
    if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
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
