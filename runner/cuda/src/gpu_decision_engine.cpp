#include "sepal/cuda/gpu_decision_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <deque>
#include <numeric>
#include <unordered_set>

#include <cuda_runtime.h>

#include "device.hpp"

namespace sepal::cuda {

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

void embedding_row_callback(const void* context, std::uint32_t token, float* out) {
    static_cast<const TrainedModel*>(context)->embedding_row(token, out);
}

constexpr std::uint64_t kFnvBasis = 1469598103934665603ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

/// h[i] = hash of tokens [0, i).
std::vector<std::uint64_t> prefix_hashes(const std::vector<std::uint32_t>& tokens) {
    std::vector<std::uint64_t> h(tokens.size() + 1);
    h[0] = kFnvBasis;
    for (std::size_t i = 0; i < tokens.size(); ++i) h[i + 1] = (h[i] ^ tokens[i]) * kFnvPrime;
    return h;
}

/// A stored prefix: the trunk's snapshot at its end and the head's memory rows for its tokens.
struct CacheEntry {
    std::vector<std::uint32_t> tokens;
    std::uint64_t hash = 0;
    DeviceSnapshot snapshot;
    DeviceBuffer<float> memory; // [tokens, width]
    std::uint64_t last_use = 0;
};

/// One row of the trunk: tokens [begin, end) of a request of the group.
struct Segment {
    std::size_t request = 0;
    std::size_t begin = 0;
    std::size_t end = 0;
    bool cache_cut = false; // leaves a snapshot that goes into the prefix cache
    std::size_t call = 0;
    std::size_t row = 0;    // index within its call
};

struct RequestPlan {
    std::shared_ptr<CacheEntry> hit;
    std::size_t hit_len = 0;
    std::size_t cut = 0; // 0: no new entry
    std::vector<std::size_t> segments; // indices into the group's segment list, in order
    DeviceSnapshot cut_snapshot;
};

std::size_t common_prefix(const std::vector<std::uint32_t>& a, const std::vector<std::uint32_t>& b) {
    const std::size_t n = std::min(a.size(), b.size());
    std::size_t i = 0;
    while (i < n && a[i] == b[i]) ++i;
    return i;
}

} // namespace

struct GpuDecisionEngine::Impl {
    TrainedModel model;
    GpuDecisionOptions options;
    std::unique_ptr<CudaEngine> trunk;
    std::unique_ptr<CudaClefHead> head;
    ScoreTimes times;
    std::uint64_t vocab = 0;

    std::vector<std::shared_ptr<CacheEntry>> cache;
    std::deque<std::vector<std::uint32_t>> recent;
    std::uint64_t clock = 0;

    Impl(const TrainedModel& m, const GpuDecisionOptions& o) : model(m), options(o) {
        if (o.profile != "gpu-f16" && o.profile != "gpu-f32") throw std::invalid_argument("unknown GPU profile '" + o.profile + "' (gpu-f16 or gpu-f32)");
        vocab = model.config().vocab_size;
        check(cudaSetDevice(o.device), "cudaSetDevice");
        EngineOptions eo;
        eo.device = o.device;
        eo.precision = o.profile == "gpu-f16" ? Precision::F16 : Precision::F32;
        eo.prune_last_layer = false; // the head reads every position
        eo.max_call_tokens = o.max_call_tokens;
        eo.max_call_rows = o.max_call_rows;
        eo.multiply_path = o.multiply_path;
        eo.fp16_accumulate = o.fp16_accumulate;
        eo.half_state = o.half_state;
        eo.gemm_rows = o.gemm_rows;
        const std::size_t head_bytes = clef_head_device_bytes(model.head_config(), o.head);
        if (o.trunk_budget_bytes != 0) {
            eo.memory_budget_bytes = o.trunk_budget_bytes;
        } else {
            std::size_t free_bytes = 0, total = 0;
            check(cudaMemGetInfo(&free_bytes, &total), "cudaMemGetInfo");
            const std::size_t usable = static_cast<std::size_t>(static_cast<double>(free_bytes) * 0.9);
            if (usable <= head_bytes + (std::size_t{512} << 20)) throw std::runtime_error("not enough free device memory for the head and the trunk");
            eo.memory_budget_bytes = usable - head_bytes;
        }
        {
            Weights<float> merged = model.merged_weights<float>();
            trunk = std::make_unique<CudaEngine>(model.config(), merged, eo);
        }
        head = std::make_unique<CudaClefHead>(model.head(), o.head, trunk->stream_handle());
    }

    std::shared_ptr<CacheEntry> find_hit(const std::vector<std::uint32_t>& tokens, const std::vector<std::uint64_t>& hashes, std::size_t limit);
    std::size_t choose_cut(const std::vector<std::uint32_t>& tokens, std::size_t limit, std::size_t hit_len, const std::deque<std::vector<std::uint32_t>>& history);
    std::vector<RequestLogits> run_group(std::span<const EncodedRequest* const> group, std::uint64_t& trunk_tokens, std::uint64_t& reused, double& head_ms);
    std::vector<RequestLogits> score(std::span<const EncodedRequest> requests);
};

std::shared_ptr<CacheEntry> GpuDecisionEngine::Impl::find_hit(const std::vector<std::uint32_t>& tokens, const std::vector<std::uint64_t>& hashes, std::size_t limit) {
    std::shared_ptr<CacheEntry> best;
    for (const std::shared_ptr<CacheEntry>& e : cache) {
        const std::size_t len = e->tokens.size();
        if (len > limit || (best && len <= best->tokens.size())) continue;
        if (hashes[len] != e->hash) continue;
        if (std::memcmp(tokens.data(), e->tokens.data(), len * sizeof(std::uint32_t)) != 0) continue;
        best = e;
    }
    return best;
}

std::size_t GpuDecisionEngine::Impl::choose_cut(const std::vector<std::uint32_t>& tokens, std::size_t limit, std::size_t hit_len,
                                                const std::deque<std::vector<std::uint32_t>>& history) {
    std::size_t cut = 0;
    const std::vector<std::uint32_t>& marker = options.schema_marker;
    if (!marker.empty() && limit >= marker.size()) {
        // the last occurrence of the marker that starts at or before the limit
        for (std::size_t i = std::min(limit, tokens.size() - marker.size() + 1); i-- > 0;) {
            if (std::memcmp(tokens.data() + i, marker.data(), marker.size() * sizeof(std::uint32_t)) == 0) {
                cut = i;
                break;
            }
        }
    } else {
        std::size_t lcp = 0;
        for (const std::vector<std::uint32_t>& h : history) lcp = std::max(lcp, common_prefix(tokens, h));
        cut = std::min(lcp, limit);
    }
    if (cut < options.prefix_min_tokens || cut <= hit_len + options.prefix_min_gain) return 0;
    return cut;
}

std::vector<RequestLogits> GpuDecisionEngine::Impl::run_group(std::span<const EncodedRequest* const> group, std::uint64_t& trunk_tokens, std::uint64_t& reused,
                                                              double& head_ms) {
    const std::size_t C = options.max_call_tokens;
    const std::size_t W = head->width();
    // ---- plan: per request the cache hit, the new cut, and its segments ----
    std::vector<RequestPlan> plans(group.size());
    std::vector<Segment> segs;
    std::deque<std::vector<std::uint32_t>> history = recent;
    for (std::size_t r = 0; r < group.size(); ++r) {
        const EncodedRequest& req = *group[r];
        const std::size_t L = req.token_ids.size();
        std::size_t first_span = L - 1;
        for (const EncodedQuestion& q : req.questions) {
            first_span = std::min<std::size_t>(first_span, q.instruction.begin);
            for (const TokenSpan& s : q.options) first_span = std::min<std::size_t>(first_span, s.begin);
        }
        RequestPlan& p = plans[r];
        if (options.prefix_cache && first_span > 0) {
            const std::vector<std::uint64_t> hashes = prefix_hashes(req.token_ids);
            p.hit = find_hit(req.token_ids, hashes, first_span);
            if (p.hit) {
                p.hit_len = p.hit->tokens.size();
                p.hit->last_use = ++clock;
                reused += p.hit_len;
            }
            p.cut = choose_cut(req.token_ids, first_span, p.hit_len, history);
            if (p.cut != 0 && find_hit(req.token_ids, hashes, p.cut) && find_hit(req.token_ids, hashes, p.cut)->tokens.size() == p.cut) p.cut = 0;
            history.push_back(req.token_ids);
            if (history.size() > options.prefix_history) history.pop_front();
        }
        std::size_t at = p.hit_len;
        auto emit = [&](std::size_t end, bool cut) {
            while (end - at > C) {
                segs.push_back({r, at, at + C, false, 0, 0});
                p.segments.push_back(segs.size() - 1);
                at += C;
            }
            segs.push_back({r, at, end, cut, 0, 0});
            p.segments.push_back(segs.size() - 1);
            at = end;
        };
        if (p.cut != 0) emit(p.cut, true);
        emit(L, false);
    }
    // ---- calls: segments in order, a new call when one does not fit; a segment of C tokens fills a call by itself ----
    std::vector<std::vector<std::size_t>> calls;
    {
        std::vector<std::size_t> current;
        std::size_t tokens_in = 0;
        for (std::size_t i = 0; i < segs.size(); ++i) {
            Segment& s = segs[i];
            const std::size_t n = s.end - s.begin;
            if (!current.empty() && (tokens_in + n > C || current.size() >= options.max_call_rows)) {
                calls.push_back(std::move(current));
                current.clear();
                tokens_in = 0;
            }
            s.call = calls.size();
            s.row = current.size();
            current.push_back(i);
            tokens_in += n;
        }
        if (!current.empty()) calls.push_back(std::move(current));
    }
    // which segments must leave a snapshot: the cut of a new entry, and any segment a later call continues
    std::vector<char> keep(segs.size(), 0);
    for (const RequestPlan& p : plans) {
        for (std::size_t k = 0; k < p.segments.size(); ++k) {
            const Segment& s = segs[p.segments[k]];
            if (s.cache_cut) keep[p.segments[k]] = 1;
            if (k + 1 < p.segments.size() && segs[p.segments[k + 1]].call != s.call) keep[p.segments[k]] = 1;
        }
    }

    const auto head_start = Clock::now();
    head->begin(group, &embedding_row_callback, &model);
    for (std::size_t r = 0; r < group.size(); ++r)
        if (plans[r].hit) head->adopt(r, plans[r].hit->memory.data(), plans[r].hit_len);
    head_ms += ms_since(head_start);

    std::vector<DeviceSnapshot> pending(segs.size()); // snapshots of kept segments, by segment
    for (const std::vector<std::size_t>& call : calls) {
        std::vector<EngineRow> rows(call.size());
        std::vector<HeadSegment> hseg(call.size());
        std::size_t row_at = 0;
        for (std::size_t k = 0; k < call.size(); ++k) {
            const Segment& s = segs[call[k]];
            const RequestPlan& p = plans[s.request];
            const EncodedRequest& req = *group[s.request];
            EngineRow& row = rows[k];
            row.token_ids.assign(req.token_ids.begin() + static_cast<std::ptrdiff_t>(s.begin), req.token_ids.begin() + static_cast<std::ptrdiff_t>(s.end));
            // the parent: the cache hit for a first segment, else the segment before it (a row of this call, or its snapshot)
            const std::size_t pos = static_cast<std::size_t>(std::find(p.segments.begin(), p.segments.end(), call[k]) - p.segments.begin());
            if (pos == 0) {
                if (p.hit) row.parent = p.hit->snapshot;
            } else {
                const std::size_t prev = p.segments[pos - 1];
                if (segs[prev].call == s.call) row.parent = ParentRow{segs[prev].row};
                else row.parent = pending[prev];
            }
            row.readout_offsets.resize(s.end - s.begin);
            std::iota(row.readout_offsets.begin(), row.readout_offsets.end(), std::size_t{0});
            row.keep_snapshot = keep[call[k]] != 0;
            hseg[k] = {s.request, s.begin, s.end - s.begin, row_at};
            row_at += s.end - s.begin;
        }
        const CudaEngine::HiddenHook hook = [&](const float* hidden, std::size_t n, void*) { head->consume(hidden, n, hseg); };
        std::vector<EngineRowResult> results;
        try {
            results = trunk->run(rows, hook, false);
        } catch (const BudgetExceeded&) {
            if (cache.empty()) throw;
            cache.clear(); // entries in use stay alive through their handles; the rest returns its memory
            results = trunk->run(rows, hook, false);
        }
        trunk_tokens += row_at;
        for (std::size_t k = 0; k < call.size(); ++k)
            if (keep[call[k]]) pending[call[k]] = std::move(results[k].snapshot);
    }

    // new cache entries: the head's memory rows of the prefix are copied out before the next batch overwrites the arena
    std::vector<std::shared_ptr<CacheEntry>> fresh;
    if (options.prefix_cache) {
        for (std::size_t r = 0; r < group.size(); ++r) {
            RequestPlan& p = plans[r];
            if (p.cut == 0) continue;
            auto e = std::make_shared<CacheEntry>();
            e->tokens.assign(group[r]->token_ids.begin(), group[r]->token_ids.begin() + static_cast<std::ptrdiff_t>(p.cut));
            e->hash = prefix_hashes(e->tokens).back();
            for (const std::size_t si : p.segments)
                if (segs[si].cache_cut) e->snapshot = pending[si];
            e->memory = DeviceBuffer<float>(p.cut * W);
            check(cudaMemcpyAsync(e->memory.data(), head->memory_rows(r), p.cut * W * sizeof(float), cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(trunk->stream_handle())), "cudaMemcpyAsync");
            e->last_use = ++clock;
            fresh.push_back(std::move(e));
        }
    }
    const auto finish_start = Clock::now();
    std::vector<RequestLogits> out = head->finish();
    head_ms += ms_since(finish_start);
    for (std::shared_ptr<CacheEntry>& e : fresh) {
        const bool duplicate = std::any_of(cache.begin(), cache.end(), [&](const auto& c) { return c->hash == e->hash && c->tokens.size() == e->tokens.size(); });
        if (!duplicate && e->snapshot.valid()) cache.push_back(std::move(e));
    }
    while (cache.size() > options.prefix_entries) {
        const auto lru = std::min_element(cache.begin(), cache.end(), [](const auto& a, const auto& b) { return a->last_use < b->last_use; });
        cache.erase(lru);
    }
    for (const EncodedRequest* r : group) {
        recent.push_back(r->token_ids);
        if (recent.size() > options.prefix_history) recent.pop_front();
    }
    return out;
}

std::vector<RequestLogits> GpuDecisionEngine::Impl::score(std::span<const EncodedRequest> requests) {
    const auto start = Clock::now();
    times = ScoreTimes{};
    for (const EncodedRequest& r : requests) {
        check_request(r, vocab);
        if (r.token_ids.size() > options.max_tokens)
            throw std::invalid_argument("a request has " + std::to_string(r.token_ids.size()) + " tokens; the most is " + std::to_string(options.max_tokens));
    }
    const ClefHeadCapacity& cap = options.head;
    std::vector<RequestLogits> out(requests.size());
    std::size_t at = 0;
    std::uint64_t trunk_tokens = 0, reused = 0;
    double head_ms = 0;
    while (at < requests.size()) {
        std::vector<const EncodedRequest*> group;
        std::size_t mem = 0, fields = 0, opts = 0, opt_tokens = 0;
        std::unordered_set<std::uint32_t> unique;
        while (at + group.size() < requests.size()) {
            const EncodedRequest& r = requests[at + group.size()];
            std::size_t f = r.questions.size(), o = 0, ot = 0;
            std::unordered_set<std::uint32_t> fresh;
            for (const EncodedQuestion& q : r.questions) {
                o += q.options.size();
                for (const TokenSpan& s : q.options) {
                    ot += s.end - s.begin;
                    for (std::uint32_t t = s.begin; t < s.end; ++t)
                        if (unique.find(r.token_ids[t]) == unique.end()) fresh.insert(r.token_ids[t]);
                }
            }
            const bool fits = group.size() < cap.requests && mem + r.token_ids.size() <= cap.memory_tokens && fields + f <= cap.fields && opts + o <= cap.options &&
                              opt_tokens + ot <= cap.option_tokens && unique.size() + fresh.size() <= cap.unique_tokens;
            if (!fits) {
                if (group.empty()) throw std::invalid_argument("a single request exceeds the head's capacity (tokens, fields, options or option tokens)");
                break;
            }
            group.push_back(&r);
            mem += r.token_ids.size();
            fields += f;
            opts += o;
            opt_tokens += ot;
            unique.insert(fresh.begin(), fresh.end());
        }
        std::vector<RequestLogits> got = run_group(group, trunk_tokens, reused, head_ms);
        for (std::size_t i = 0; i < got.size(); ++i) out[at + i] = std::move(got[i]);
        at += group.size();
    }
    times.tokens = trunk_tokens;
    times.tokens_reused = reused;
    times.total_ms = ms_since(start);
    times.head_ms = head_ms;
    times.trunk_ms = times.total_ms - head_ms;
    return out;
}

GpuDecisionEngine::GpuDecisionEngine(const TrainedModel& model, const GpuDecisionOptions& options) : impl_(std::make_unique<Impl>(model, options)) {}
GpuDecisionEngine::~GpuDecisionEngine() = default;
std::vector<RequestLogits> GpuDecisionEngine::score(std::span<const EncodedRequest> requests) { return impl_->score(requests); }
std::string GpuDecisionEngine::profile() const { return impl_->options.profile; }
std::uint64_t GpuDecisionEngine::max_tokens() const { return impl_->options.max_tokens; }
ScoreTimes GpuDecisionEngine::last_times() const { return impl_->times; }
CudaEngine& GpuDecisionEngine::trunk() { return *impl_->trunk; }
const GpuDecisionOptions& GpuDecisionEngine::options() const { return impl_->options; }
EngineMemory GpuDecisionEngine::trunk_memory() const { return impl_->trunk->memory(); }

} // namespace sepal::cuda
