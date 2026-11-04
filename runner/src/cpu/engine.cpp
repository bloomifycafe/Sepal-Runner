#include "sepal/cpu/engine.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <deque>
#include <list>
#include <mutex>
#include <span>
#include <stdexcept>

#include "sepal/cpu/clef_head.hpp"
#include "sepal/cpu/gemm.hpp"

namespace sepal::cpu {

Profile parse_profile(const std::string& id) {
    if (id == "cpu-f32") return Profile::F32;
    if (id == "cpu-bf16") return Profile::BF16;
    if (id == "cpu-q8") return Profile::Q8;
    throw std::invalid_argument("unknown CPU profile '" + id + "' (cpu-f32, cpu-bf16 or cpu-q8)");
}

namespace {

class CpuDecisionEngine final : public DecisionEngine {
public:
    CpuDecisionEngine(TrainedModel model, const CpuEngineOptions& options)
        : model_(std::move(model)), options_(options), profile_(parse_profile(options.profile)) {
        TrunkOptions t;
        t.profile = profile_;
        t.q8 = options.q8;
        t.threads = options.threads;
        t.chunk_tokens = options.chunk_tokens;
        t.step_tokens = options.step_tokens;
        t.delta_chunk = options.delta_chunk;
        t.force_isa = options.force_isa;
        t.isa = options.isa;
        t.cpus = options.cpus;
        trunk_ = std::make_unique<CpuTrunk>(
            model_.config(), [this](const std::string& scope) { return model_.merged_scope<float>(scope); },
            [this](std::uint32_t token, float* out) { model_.embedding_row(token, out); }, t);
        head_ = std::make_unique<CpuClefHead>(model_.head(), trunk_->pool(), trunk_->isa());
    }

    // ---- prefix states -------------------------------------------------------------------------------------------------
    struct CacheEntry {
        std::vector<std::uint32_t> tokens;
        PrefixState state;
        std::uint64_t last_use = 0;
    };

    static std::size_t common_prefix(const std::vector<std::uint32_t>& a, const std::vector<std::uint32_t>& b) {
        const std::size_t n = std::min(a.size(), b.size());
        std::size_t i = 0;
        while (i < n && a[i] == b[i]) ++i;
        return i;
    }

    // One request through the prefix cache: resume from the longest state that is a prefix of it; make a longer state when an earlier request shares
    // more tokens than that (the tokens before the first question are the log, the questions differ); run the rest; run the head.
    RequestLogits score_with_prefix(const EncodedRequest& r, const CpuTrunk::EmbeddingFn& embed) {
        const std::vector<std::uint32_t>& ids = r.token_ids;
        std::size_t first_span = ids.size() - 1;
        for (const EncodedQuestion& q : r.questions) {
            first_span = std::min<std::size_t>(first_span, q.instruction.begin);
            for (const TokenSpan& s : q.options) first_span = std::min<std::size_t>(first_span, s.begin);
        }
        std::shared_ptr<CacheEntry> hit;
        for (const auto& e : cache_)
            if (e->tokens.size() <= first_span && (!hit || e->tokens.size() > hit->tokens.size()) &&
                std::equal(e->tokens.begin(), e->tokens.end(), ids.begin()))
                hit = e;
        const std::size_t hit_len = hit ? hit->tokens.size() : 0;
        std::size_t shared = 0;
        for (const auto& h : history_) shared = std::max(shared, common_prefix(ids, h));
        const std::size_t cut = std::min(shared, first_span);
        const std::uint64_t budget = options_.prefix_cache_mib * (1ull << 20);
        std::shared_ptr<CacheEntry> use = hit;
        if (cut >= options_.prefix_min_tokens && cut >= hit_len + options_.prefix_min_gain) {
            auto fresh = std::make_shared<CacheEntry>();
            fresh->tokens.assign(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(cut));
            std::vector<std::vector<float>> scratch;
            const std::vector<std::span<const std::uint32_t>> one{std::span<const std::uint32_t>(ids.data(), cut)};
            const std::vector<Resume> resume{Resume{hit ? &hit->state : nullptr, &fresh->state}};
            trunk_->forward(one, scratch, nullptr, &resume);
            times_.trunk_ms += trunk_->last_times().total_ms;
            times_.tokens += cut - hit_len;
            times_.tokens_reused += hit_len;
            fresh->last_use = ++clock_;
            const std::uint64_t bytes = fresh->state.bytes();
            if (bytes <= budget) {
                cache_bytes_ += bytes;
                cache_.push_back(fresh);
                while (cache_bytes_ > budget && cache_.size() > 1) {
                    const auto lru = std::min_element(cache_.begin(), cache_.end(), [](const auto& a, const auto& b) { return a->last_use < b->last_use; });
                    if (*lru == fresh) break;
                    cache_bytes_ -= (*lru)->state.bytes();
                    cache_.erase(lru);
                }
            }
            use = fresh;
        } else if (hit) {
            hit->last_use = ++clock_;
        }
        std::vector<std::vector<float>> hidden;
        const std::vector<std::span<const std::uint32_t>> whole{std::span<const std::uint32_t>(ids)};
        const std::vector<Resume> resume{Resume{use ? &use->state : nullptr, nullptr}};
        trunk_->forward(whole, hidden, nullptr, &resume);
        const std::size_t reused = use ? use->tokens.size() : 0;
        times_.trunk_ms += trunk_->last_times().total_ms;
        times_.tokens += ids.size() - reused;
        times_.tokens_reused += reused;
        history_.push_back(ids);
        if (history_.size() > 8) history_.pop_front();
        const auto head_start = std::chrono::steady_clock::now();
        RequestLogits out = head_->run(hidden[0], r, embed);
        times_.head_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - head_start).count();
        return out;
    }

    std::vector<RequestLogits> score(std::span<const EncodedRequest> requests) override {
        std::lock_guard<std::mutex> lock(mutex_);
        const std::uint64_t vocab = model_.config().vocab_size;
        for (const EncodedRequest& r : requests) {
            check_request(r, vocab);
            if (r.token_ids.size() > options_.max_tokens) {
                throw std::invalid_argument("request of " + std::to_string(r.token_ids.size()) + " tokens exceeds max_tokens " +
                                            std::to_string(options_.max_tokens));
            }
        }
        times_ = {};
        const auto start = std::chrono::steady_clock::now();
        std::vector<RequestLogits> results(requests.size());
        const auto embed = [this](std::uint32_t token, float* out) { model_.embedding_row(token, out); };
        if (options_.prefix_cache_mib != 0) {      // one request at a time through the prefix cache; the grouped path is for the cache-off case
            for (std::size_t k = 0; k < requests.size(); ++k) results[k] = score_with_prefix(requests[k], embed);
            times_.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            return results;
        }
        std::size_t i = 0;
        while (i < requests.size()) {
            std::size_t j = i, tokens = 0;
            while (j < requests.size() && (j == i || tokens + requests[j].token_ids.size() <= options_.group_tokens)) {
                tokens += requests[j].token_ids.size();
                ++j;
            }
            std::vector<std::span<const std::uint32_t>> spans;
            for (std::size_t k = i; k < j; ++k) spans.emplace_back(requests[k].token_ids);
            std::vector<std::vector<float>> hidden;
            trunk_->forward(spans, hidden);
            times_.trunk_ms += trunk_->last_times().total_ms;
            times_.tokens += tokens;
            const auto head_start = std::chrono::steady_clock::now();
            for (std::size_t k = i; k < j; ++k) {
                results[k] = head_->run(hidden[k - i], requests[k], embed);
                std::vector<float>().swap(hidden[k - i]);
            }
            times_.head_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - head_start).count();
            i = j;
        }
        times_.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        return results;
    }

    std::string profile() const override { return options_.profile; }
    std::string runtime() const override {
        const Isa isa = trunk_->isa();
        const Variant v = pick_variant(profile_ == Profile::F32 ? WeightKind::F32 : WeightKind::BF16, isa);
        const GemmBlocking b = gemm_blocking(v);
        const CacheSizes& cs = cache_sizes();
        return std::string("kernels ") + name_of(isa) + " (" + name_of(v) + " " + std::to_string(info(v).mr) + "x" + std::to_string(info(v).nr) +
               ", kc " + std::to_string(b.kc) + ", rows " + std::to_string(b.rows) + ", cols " + std::to_string(b.cols) + "; L1D " +
               std::to_string(cs.l1d / 1024) + " KiB, L2 " + std::to_string(cs.l2 / 1024) + " KiB, L3 " + std::to_string(cs.l3 / 1024) +
               " KiB) on " + features().brand + "; " + trunk_->pool().layout();
    }
    std::uint64_t max_tokens() const override { return options_.max_tokens; }
    ScoreTimes last_times() const override { return times_; }

private:
    TrainedModel model_;
    CpuEngineOptions options_;
    Profile profile_;
    std::unique_ptr<CpuTrunk> trunk_;
    std::unique_ptr<CpuClefHead> head_;
    std::mutex mutex_;
    ScoreTimes times_;
    std::list<std::shared_ptr<CacheEntry>> cache_;
    std::deque<std::vector<std::uint32_t>> history_;
    std::uint64_t cache_bytes_ = 0;
    std::uint64_t clock_ = 0;
};

} // namespace

std::unique_ptr<DecisionEngine> make_cpu_engine(TrainedModel model, const CpuEngineOptions& options) {
    return std::make_unique<CpuDecisionEngine>(std::move(model), options);
}

std::unique_ptr<DecisionEngine> make_cpu_engine(const std::string& final_dir, const std::string& base_dir, const CpuEngineOptions& options) {
    return make_cpu_engine(TrainedModel::open(final_dir, base_dir), options);
}

} // namespace sepal::cpu
