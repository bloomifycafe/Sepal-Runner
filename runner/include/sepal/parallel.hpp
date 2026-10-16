// A minimal fork-join loop over [0, n): used by load-time work (the LoRA merge). Not a thread pool.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace sepal {

/// Calls fn(i) for every i in [0, n) on up to `threads` threads (0: hardware concurrency). Work is handed out in
/// chunks of `chunk` indices. The first exception thrown by fn is rethrown after all threads have stopped.
template <class Fn>
void parallel_for(std::size_t n, Fn&& fn, std::size_t chunk = 1, unsigned threads = 0) {
    if (n == 0) return;
    if (chunk == 0) chunk = 1;
    unsigned count = threads != 0 ? threads : std::thread::hardware_concurrency();
    if (count == 0) count = 1;
    const std::size_t chunks = (n + chunk - 1) / chunk;
    count = static_cast<unsigned>(std::min<std::size_t>(count, chunks));
    std::atomic<std::size_t> next{0};
    std::exception_ptr error;
    std::mutex error_mutex;
    const auto work = [&] {
        for (;;) {
            const std::size_t c = next.fetch_add(1);
            if (c >= chunks) return;
            const std::size_t begin = c * chunk;
            const std::size_t end = std::min(n, begin + chunk);
            try {
                for (std::size_t i = begin; i < end; ++i) fn(i);
            } catch (...) {
                const std::lock_guard<std::mutex> lock(error_mutex);
                if (!error) error = std::current_exception();
                next.store(chunks);
                return;
            }
        }
    };
    std::vector<std::thread> pool;
    for (unsigned t = 1; t < count; ++t) pool.emplace_back(work);
    work();
    for (std::thread& t : pool) t.join();
    if (error) std::rethrow_exception(error);
}

} // namespace sepal
