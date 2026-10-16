#include "sepal/cpu/engine.hpp"

#include <chrono>
#include <mutex>
#include <span>
#include <stdexcept>

#include "sepal/cpu/clef_head.hpp"

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
        {
            const Weights<float> merged = model_.merged_weights<float>();
            trunk_ = std::make_unique<CpuTrunk>(
                model_.config(), merged, [this](std::uint32_t token, float* out) { model_.embedding_row(token, out); }, t);
        }
        head_ = std::make_unique<CpuClefHead>(model_.head(), trunk_->pool(), trunk_->isa());
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
};

} // namespace

std::unique_ptr<DecisionEngine> make_cpu_engine(TrainedModel model, const CpuEngineOptions& options) {
    return std::make_unique<CpuDecisionEngine>(std::move(model), options);
}

std::unique_ptr<DecisionEngine> make_cpu_engine(const std::string& final_dir, const std::string& base_dir, const CpuEngineOptions& options) {
    return make_cpu_engine(TrainedModel::open(final_dir, base_dir), options);
}

} // namespace sepal::cpu
