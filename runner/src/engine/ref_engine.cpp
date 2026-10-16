// "cpu-ref" / "cpu-ref64": the C++ oracle. The LoRA is merged in float32 (float64 for cpu-ref64), the trunk runs on the
// sepal::ref executor (single-threaded, scalar), the final-normed hidden state of EVERY position is read, and the
// reference head turns it into logits. Slow by design; every faster engine is compared with this one.
#include <chrono>
#include <numeric>

#include "sepal/clef_head.hpp"
#include "sepal/engine.hpp"
#include "sepal/ref/executor.hpp"
#include "sepal/trained_model.hpp"

namespace sepal {

namespace {

void embedding_callback(const void* context, std::uint32_t token, float* out) {
    static_cast<const TrainedModel*>(context)->embedding_row(token, out);
}

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

template <class T>
class RefEngine final : public DecisionEngine {
public:
    RefEngine(TrainedModel model, std::uint64_t max_tokens, std::string profile)
        : model_(std::move(model)), weights_(model_.merged_weights<T>()), gemm_(make_reference_gemm()),
          max_tokens_(max_tokens), profile_(std::move(profile)) {}

    std::vector<RequestLogits> score(std::span<const EncodedRequest> requests) override {
        const auto start = std::chrono::steady_clock::now();
        for (const EncodedRequest& request : requests) {
            check_request(request, model_.config().vocab_size);
            if (request.token_ids.size() > max_tokens_) {
                throw std::invalid_argument("request has " + std::to_string(request.token_ids.size()) + " tokens, the limit is " +
                                            std::to_string(max_tokens_));
            }
        }
        times_ = {};
        const std::size_t hidden_size = model_.config().hidden_size;
        std::vector<RequestLogits> results;
        for (const EncodedRequest& request : requests) {
            const auto trunk_start = std::chrono::steady_clock::now();
            ref::Row<T> row;
            row.token_ids = request.token_ids;
            row.readout_offsets.resize(row.token_ids.size());
            std::iota(row.readout_offsets.begin(), row.readout_offsets.end(), std::size_t{0});
            const std::vector<ref::Row<T>> rows = {std::move(row)};
            const std::vector<ref::RowResult<T>> out = ref::run_rows<T>(*gemm_, model_.config(), weights_, rows);
            std::vector<float> hidden(out[0].hidden.size());
            for (std::size_t i = 0; i < hidden.size(); ++i) hidden[i] = static_cast<float>(out[0].hidden[i]);
            times_.trunk_ms += ms_since(trunk_start);
            times_.tokens += request.token_ids.size();
            const auto head_start = std::chrono::steady_clock::now();
            results.push_back(clef_head_reference(model_.head(), hidden, request, &embedding_callback, &model_));
            times_.head_ms += ms_since(head_start);
            (void)hidden_size;
        }
        times_.total_ms = ms_since(start);
        return results;
    }

    std::string profile() const override { return profile_; }
    std::uint64_t max_tokens() const override { return max_tokens_; }
    std::uint64_t hidden_size() const override { return model_.config().hidden_size; }
    ScoreTimes last_times() const override { return times_; }

private:
    TrainedModel model_;
    Weights<T> weights_;
    std::unique_ptr<GemmProvider> gemm_;
    std::uint64_t max_tokens_;
    std::string profile_;
    ScoreTimes times_;
};

} // namespace

std::unique_ptr<DecisionEngine> make_ref_engine(const EngineOptions& options, bool fp64) {
    if (options.max_tokens == 0) throw std::invalid_argument("max_tokens must be positive");
    TrainedModel model = TrainedModel::open(options.final_dir, options.base_dir);
    if (fp64) return std::make_unique<RefEngine<double>>(std::move(model), options.max_tokens, "cpu-ref64");
    return std::make_unique<RefEngine<float>>(std::move(model), options.max_tokens, "cpu-ref");
}

} // namespace sepal
