// Suites `head_fixture` (tiny geometry against PyTorch float64) and `head_golden` (the golden set of the trained model,
// head only: hidden_fp32 in, logits_fp32 out). The real-model suite prints "SKIPPED: real-model ..." unless
// SEPAL_GOLDEN_DIR, SEPAL_TRAINED_FINAL and SEPAL_TRAINED_BASE are set.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>

#include "head_case.hpp"
#include "sepal/clef_head.hpp"
#include "sepal/cuda/clef_head_cuda.hpp"
#include "sepal/json.hpp"
#include "sepal/trained_model.hpp"
#include "device.hpp"
#include "test.hpp"

#ifndef SEPAL_FIXTURE_DIR
#error "SEPAL_FIXTURE_DIR must be defined by the build"
#endif

using namespace sepal;
using namespace sepal::cuda;
namespace fs = std::filesystem;
using sepal::test::env_value;

namespace {

using Clock = std::chrono::steady_clock;

std::string tiny_final() { return std::string(SEPAL_FIXTURE_DIR) + "/tiny_trained"; }
std::string tiny_base() { return std::string(SEPAL_FIXTURE_DIR) + "/tiny_checkpoint"; }

std::string read_text(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void embedding_callback(const void* context, std::uint32_t token, float* out) {
    static_cast<const TrainedModel*>(context)->embedding_row(token, out);
}

ClefHeadCapacity small_capacity() {
    ClefHeadCapacity c;
    c.memory_tokens = 8192;
    c.call_tokens = 4096;
    c.requests = 16;
    c.fields = 128;
    c.options = 512;
    c.option_tokens = 16384;
    c.unique_tokens = 2048;
    c.score_floats = std::size_t{1} << 22;
    return c;
}

std::vector<RequestLogits> run_head(CudaClefHead& head, const std::vector<const EncodedRequest*>& reqs, const std::vector<const std::vector<float>*>& hidden,
                                    const TrainedModel& m, std::size_t chunk) {
    head.begin(reqs, &embedding_callback, &m);
    const std::size_t hd = head.config().hidden_size;
    for (std::size_t r = 0; r < reqs.size(); ++r) {
        const std::size_t L = reqs[r]->token_ids.size();
        for (std::size_t first = 0; first < L; first += chunk) {
            const std::size_t n = std::min(chunk, L - first);
            DeviceBuffer<float> d(n * hd);
            d.upload(hidden[r]->data() + first * hd, n * hd);
            const HeadSegment seg{r, first, n, 0};
            head.consume(d.data(), n, std::span<const HeadSegment>(&seg, 1));
            check(cudaStreamSynchronize(nullptr), "sync"); // the uploaded buffer goes out of scope
        }
    }
    return head.finish();
}

} // namespace

SEPAL_TEST(head_fixture, matches_pytorch_float64) {
    const TrainedModel m = TrainedModel::open(tiny_final(), tiny_base());
    const sepal::test::GoldenCase c = sepal::test::read_case(tiny_final() + "/head_case.safetensors");
    cudaStream_t stream = nullptr;
    check(cudaStreamCreate(&stream), "cudaStreamCreate");
    {
        CudaClefHead head(m.head(), small_capacity(), stream);
        const std::size_t L = c.request.token_ids.size();
        for (const std::size_t chunk : {L, std::size_t{7}, std::size_t{1}}) {
            const auto out = run_head(head, {&c.request}, {&c.hidden}, m, chunk);
            const sepal::test::Parity parity = sepal::test::compare_logits(out[0], c.logits);
            std::cout << "  chunk " << chunk << ": max |dlogit| " << std::scientific << parity.max_dlogit << " max |dp| " << parity.max_dp << "\n";
            CHECK(parity.max_dlogit <= 2e-5);
            CHECK_EQ(parity.flips, std::size_t{0});
        }
        // a batch of several requests gives each one its own answer
        const auto out = run_head(head, {&c.request, &c.request, &c.request}, {&c.hidden, &c.hidden, &c.hidden}, m, L);
        CHECK_EQ(out.size(), std::size_t{3});
        for (const RequestLogits& o : out) {
            const sepal::test::Parity parity = sepal::test::compare_logits(o, c.logits);
            CHECK(parity.max_dlogit <= 2e-5);
        }
        // bad requests are refused and the head stays usable
        EncodedRequest bad = c.request;
        bad.questions[0].options[0] = {0, static_cast<std::uint32_t>(L + 1)};
        const EncodedRequest* bp = &bad;
        CHECK_THROWS(head.begin(std::span<const EncodedRequest* const>(&bp, 1), &embedding_callback, &m), std::invalid_argument);
        const auto again = run_head(head, {&c.request}, {&c.hidden}, m, L);
        CHECK(sepal::test::compare_logits(again[0], c.logits).max_dlogit <= 2e-5);
    }
    check(cudaStreamDestroy(stream), "cudaStreamDestroy");
}

SEPAL_TEST(head_golden, golden_set) {
    const auto golden = env_value("SEPAL_GOLDEN_DIR");
    if (!golden || !sepal::test::real_model_available("head golden")) {
        if (!golden) std::cout << "SKIPPED: real-model head golden test; set SEPAL_GOLDEN_DIR (runs/runner-golden/full08b)\n";
        return;
    }
    const TrainedModel m = TrainedModel::open(*env_value("SEPAL_TRAINED_FINAL"), *env_value("SEPAL_TRAINED_BASE"));
    const json::Value manifest = json::parse(read_text(fs::path(*golden) / "manifest.json"));
    std::uint64_t max_tokens = 40000;
    if (const auto limit = env_value("SEPAL_GOLDEN_MAX_TOKENS")) max_tokens = std::stoull(*limit);
    cudaStream_t stream = nullptr;
    check(cudaStreamCreate(&stream), "cudaStreamCreate");
    ClefHeadCapacity cap;
    cap.memory_tokens = 65536;
    cap.call_tokens = 16384;
    double worst_logit = 0, worst_p = 0;
    std::size_t flips = 0, run = 0;
    {
    CudaClefHead head(m.head(), cap, stream);
    for (const json::Value& entry : manifest.at("cases").items()) {
        if (entry.at("tokens").as_uint() > max_tokens) continue;
        const std::string file = (fs::path(*golden) / entry.at("file").as_string()).string();
        const sepal::test::GoldenCase c = sepal::test::read_case(file);
        const auto start = Clock::now();
        const auto out = run_head(head, {&c.request}, {&c.hidden}, m, cap.call_tokens);
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        const sepal::test::Parity parity = sepal::test::compare_logits(out[0], c.logits);
        std::cout << "  " << entry.at("name").as_string() << ": L=" << c.request.token_ids.size() << " max |dlogit| " << std::scientific << parity.max_dlogit
                  << " max |dp| " << parity.max_dp << " flips " << parity.flips << " (" << std::fixed << std::setprecision(1) << ms << " ms)\n";
        worst_logit = std::fmax(worst_logit, parity.max_dlogit);
        worst_p = std::fmax(worst_p, parity.max_dp);
        flips += parity.flips;
        ++run;
    }
    } // the head goes before its stream
    std::cout << "  golden head (GPU, fp32): " << run << " case(s), max |dlogit| " << std::scientific << worst_logit << ", max |dp| " << worst_p << ", flips " << flips << "\n";
    CHECK(run > 0);
    CHECK(worst_logit <= 1e-4);
    CHECK(worst_p <= 1e-5);
    CHECK_EQ(flips, std::size_t{0});
    check(cudaStreamDestroy(stream), "cudaStreamDestroy");
}
