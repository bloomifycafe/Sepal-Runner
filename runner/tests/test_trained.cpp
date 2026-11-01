// Suites `trained` (LoRA merge, adapter validation, embedding row) and `clef` (head config, weights, the reference
// head against the oracle). The fixture in tests/fixtures/tiny_trained is written by tools/make_trained_fixture.py:
// its expected merges come from NumPy and its expected logits from the Clef head run by PyTorch in float64.
// The real-model tests skip unless SEPAL_TRAINED_FINAL / SEPAL_TRAINED_BASE (and SEPAL_GOLDEN_DIR) are set.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>

#include "head_case.hpp"
#include "sepal/clef_head.hpp"
#include "sepal/json.hpp"
#include "sepal/trained_model.hpp"
#include "test.hpp"

#ifndef SEPAL_FIXTURE_DIR
#error "SEPAL_FIXTURE_DIR must be defined by the build"
#endif

using namespace sepal;
namespace fs = std::filesystem;
using sepal::test::env_value;

namespace {

using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

std::string tiny_final() { return std::string(SEPAL_FIXTURE_DIR) + "/tiny_trained"; }
std::string tiny_base() { return std::string(SEPAL_FIXTURE_DIR) + "/tiny_checkpoint"; }

std::string read_text(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void embedding_callback(const void* context, std::uint32_t token, float* out) {
    static_cast<const TrainedModel*>(context)->embedding_row(token, out);
}

// A copy of the fixture's final/ directory in a temporary place, for tests that break one file.
struct FinalCopy {
    fs::path dir;
    FinalCopy() {
        static int counter = 0;
        dir = fs::temp_directory_path() / ("sepal_trained_test_" + std::to_string(++counter));
        fs::remove_all(dir);
        fs::copy(fs::path(tiny_final()), dir, fs::copy_options::recursive);
    }
    ~FinalCopy() {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
    void rewrite_config(const std::string& from, const std::string& to) const {
        std::string text = read_text(dir / "adapter_config.json");
        const std::size_t at = text.find(from);
        if (at == std::string::npos) throw std::logic_error("fixture adapter_config has no '" + from + "'");
        text.replace(at, from.size(), to);
        std::ofstream(dir / "adapter_config.json", std::ios::binary) << text;
    }
    std::string str() const { return dir.string(); }
};

std::string describe(const std::exception& e) { return e.what(); }

} // namespace

SEPAL_TEST(trained, fixture_opens) {
    const TrainedModel m = TrainedModel::open(tiny_final(), tiny_base());
    CHECK_EQ(m.lora().rank, 4u);
    CHECK_NEAR(m.lora().alpha, 8.0, 0.0);
    CHECK_NEAR(m.lora().scale(), 2.0, 0.0);
    CHECK_EQ(m.lora().modules, std::size_t{6});
    CHECK_EQ(m.head_config().width, 16u);
    CHECK_EQ(m.config().hidden_size, std::uint64_t{24});
    CHECK(!m.sepal_json().empty());
    CHECK(m.load_seconds() >= 0.0);
}

SEPAL_TEST(trained, merge_matches_numpy) {
    const TrainedModel m = TrainedModel::open(tiny_final(), tiny_base());
    const SafetensorsFile expected = SafetensorsFile::open(tiny_final() + "/expected_merged.safetensors");
    const Weights<float> w32 = m.merged_weights<float>();
    const Weights<double> w64 = m.merged_weights<double>();
    const Weights<float> base32 = m.base().weights<float>();
    std::size_t adapted = 0;
    double worst32 = 0, worst64 = 0;
    for (const TensorView& t : expected.tensors()) {
        const bool is32 = t.name.starts_with("f32.");
        const std::string name = t.name.substr(4);
        ++adapted;
        if (is32) {
            const auto want = t.values<float>();
            const BasicTensor<float>& got = w32[name];
            CHECK_EQ(got.size(), want.size());
            for (std::size_t i = 0; i < want.size(); ++i) worst32 = std::fmax(worst32, std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i])));
            // the merge changed it
            CHECK(std::fabs(static_cast<double>(got[0]) - static_cast<double>(base32[name][0])) > 0.0 || std::fabs(static_cast<double>(got[1]) - static_cast<double>(base32[name][1])) > 0.0);
        } else {
            const auto want = t.values<double>();
            const BasicTensor<double>& got = w64[name];
            CHECK_EQ(got.size(), want.size());
            for (std::size_t i = 0; i < want.size(); ++i) worst64 = std::fmax(worst64, std::fabs(got[i] - want[i]));
        }
    }
    CHECK_EQ(adapted, std::size_t{12});
    std::cout << "  merge vs NumPy: max |diff| float32 " << worst32 << ", float64 " << worst64 << "\n";
    CHECK(worst32 <= 1e-7);
    CHECK(worst64 <= 1e-12);
    // everything unadapted is the base, exactly (norms, embedding, other projections)
    for (const char* name : {"embed_tokens.weight", "norm.weight", "layers.0.mlp.up_proj.weight", "layers.3.self_attn.k_proj.weight"}) {
        const BasicTensor<float>& a = w32[name];
        const BasicTensor<float>& b = base32[name];
        bool same = a.size() == b.size();
        for (std::size_t i = 0; same && i < a.size(); ++i) same = a[i] == b[i];
        CHECK(same);
    }
}

SEPAL_TEST(trained, streamed_merge_equals_whole_merge) {
    // merged_tensor and merged_scope (what the engines use to avoid holding the whole model in float32) give the same bits as merged_weights.
    const TrainedModel m = TrainedModel::open(tiny_final(), tiny_base());
    const Weights<float> whole = m.merged_weights<float>();
    const auto same = [](const BasicTensor<float>& a, const BasicTensor<float>& b) {
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i)
            if (a[i] != b[i]) return false;
        return true;
    };
    for (const char* name : {"embed_tokens.weight", "norm.weight", "layers.0.mlp.up_proj.weight", "layers.3.self_attn.k_proj.weight"})
        CHECK(same(m.merged_tensor<float>(name), whole[name]));
    const Weights<float> layer = m.merged_scope<float>("layers.0");
    for (const char* name : {"mlp.up_proj.weight", "mlp.down_proj.weight", "input_layernorm.weight"})
        CHECK(same(layer[name], whole.scope("layers.0")[name]));
    CHECK_EQ(m.merged_scope<float>("norm")["weight"].size(), whole["norm.weight"].size());
}

SEPAL_TEST(trained, embedding_row_is_the_base_row) {
    const TrainedModel m = TrainedModel::open(tiny_final(), tiny_base());
    const Weights<float> base = m.base().weights<float>();
    std::vector<float> row(m.config().hidden_size);
    for (const std::uint32_t token : {0u, 17u, 63u}) {
        m.embedding_row(token, row.data());
        bool same = true;
        for (std::size_t i = 0; i < row.size(); ++i) same = same && row[i] == base["embed_tokens.weight"][token * row.size() + i];
        CHECK(same);
    }
    CHECK_THROWS(m.embedding_row(64, row.data()), std::out_of_range);
}

SEPAL_TEST(trained, refuses_unsupported_adapters) {
    const struct { const char* from; const char* to; const char* what; } cases[] = {
        {"\"use_dora\": false", "\"use_dora\": true", "use_dora"},
        {"\"use_rslora\": false", "\"use_rslora\": true", "use_rslora"},
        {"\"bias\": \"none\"", "\"bias\": \"all\"", "bias"},
        {"\"rank_pattern\": {}", "\"rank_pattern\": {\"layers.0.mlp.down_proj\": 8}", "rank_pattern"},
        {"\"alpha_pattern\": {}", "\"alpha_pattern\": {\"x\": 8}", "alpha_pattern"},
        {"\"modules_to_save\": null", "\"modules_to_save\": [\"embed_tokens\"]", "modules_to_save"},
        {"\"r\": 4", "\"r\": 8", "wrong shape"}, // lora_A is [4, in]
        {"\"peft_type\": \"LORA\"", "\"peft_type\": \"IA3\"", "peft_type"},
    };
    for (const auto& c : cases) {
        const FinalCopy copy;
        copy.rewrite_config(c.from, c.to);
        bool threw = false;
        try {
            (void)TrainedModel::open(copy.str(), tiny_base());
        } catch (const std::exception& e) {
            threw = true;
            const std::string message = describe(e);
            CHECK(message.find(c.what) != std::string::npos);
            if (message.find(c.what) == std::string::npos) std::cerr << "  message was: " << message << "\n";
        }
        CHECK(threw);
    }
}

SEPAL_TEST(trained, refuses_missing_files_and_bad_base) {
    CHECK_THROWS(TrainedModel::open("/no/such/dir", tiny_base()), std::exception);
    CHECK_THROWS(TrainedModel::open(tiny_final(), "/no/such/base"), std::exception);
    // a head whose hidden size is not the base's
    const FinalCopy copy;
    std::string text = read_text(copy.dir / "joint_head_config.json");
    const std::size_t at = text.find("24");
    text.replace(at, 2, "32");
    std::ofstream(copy.dir / "joint_head_config.json", std::ios::binary) << text;
    CHECK_THROWS(TrainedModel::open(copy.str(), tiny_base()), std::exception);
}

SEPAL_TEST(trained_real, real_model) {
    if (!sepal::test::real_model_available("trained")) return;
    const auto start = Clock::now();
    const TrainedModel m = TrainedModel::open(*env_value("SEPAL_TRAINED_FINAL"), *env_value("SEPAL_TRAINED_BASE"));
    std::cout << "  open: " << std::fixed << std::setprecision(2) << seconds_since(start) << " s (internal " << m.load_seconds() << " s)\n";
    CHECK_EQ(m.lora().rank, 64u);
    CHECK_NEAR(m.lora().alpha, 128.0, 0.0);
    CHECK_EQ(m.lora().modules, std::size_t{186});
    CHECK_EQ(m.head().tensors.size(), std::size_t{122});
    CHECK_EQ(m.config().hidden_size, std::uint64_t{1024});
    const auto merge_start = Clock::now();
    const Weights<float> w = m.merged_weights<float>();
    std::cout << "  merged_weights<float>: " << seconds_since(merge_start) << " s\n";
    // the merge changed an adapted weight and left the embedding alone
    const Weights<float> base = m.base().weights<float>();
    const char* name = "layers.0.mlp.down_proj.weight";
    double diff = 0;
    for (std::size_t i = 0; i < w[name].size(); ++i) diff = std::fmax(diff, std::fabs(static_cast<double>(w[name][i]) - static_cast<double>(base[name][i])));
    CHECK(diff > 0.0);
    CHECK(w["embed_tokens.weight"][12345] == base["embed_tokens.weight"][12345]);
    std::vector<float> row(1024);
    m.embedding_row(777, row.data());
    CHECK(row[5] == base["embed_tokens.weight"][777 * 1024 + 5]);
}

SEPAL_TEST(clef, config_parse) {
    const ClefHeadConfig c = ClefHeadConfig::parse(R"({"hidden_size":1024,"width":512,"routing_layers":2,"layers":4,"heads":8,"feedforward":2048})");
    CHECK_EQ(c.hidden_size, 1024u);
    CHECK_EQ(c.width, 512u);
    CHECK_EQ(c.routing_layers, 2u);
    CHECK_EQ(c.layers, 4u);
    CHECK_EQ(c.heads, 8u);
    CHECK_EQ(c.feedforward, 2048u);
    CHECK_THROWS(ClefHeadConfig::parse("{}"), std::invalid_argument);
    CHECK_THROWS(ClefHeadConfig::parse("not json"), std::invalid_argument);
    CHECK_THROWS(ClefHeadConfig::parse(R"({"hidden_size":1024,"width":500,"routing_layers":2,"layers":4,"heads":8,"feedforward":2048})"), std::invalid_argument);
    CHECK_THROWS(ClefHeadConfig::parse(R"({"hidden_size":1024,"width":512,"routing_layers":2,"layers":4,"heads":8,"feedforward":2048,"extra":1})"), std::invalid_argument);
    CHECK_THROWS(ClefHeadConfig::parse(R"({"hidden_size":-1,"width":512,"routing_layers":2,"layers":4,"heads":8,"feedforward":2048})"), std::invalid_argument);
}

SEPAL_TEST(clef, weights_strict_names_and_shapes) {
    const ClefHeadConfig c = ClefHeadConfig::parse(read_text(tiny_final() + "/joint_head_config.json"));
    const ClefHeadWeights w = ClefHeadWeights::open(tiny_final() + "/joint_head.safetensors", c);
    CHECK_EQ(w.tensors.size(), w.shapes.size());
    CHECK(w.shapes.at("prior_logit_scale").empty());
    CHECK_EQ(w.at("prior_logit_scale").size(), std::size_t{1});
    CHECK_NEAR(w.at("prior_logit_scale")[0], 1.5, 1e-7);
    // a config with one more layer expects tensors the file does not have
    ClefHeadConfig more = c;
    more.layers += 1;
    CHECK_THROWS(ClefHeadWeights::open(tiny_final() + "/joint_head.safetensors", more), std::runtime_error);
    ClefHeadConfig fewer = c;
    fewer.layers -= 1;
    CHECK_THROWS(ClefHeadWeights::open(tiny_final() + "/joint_head.safetensors", fewer), std::runtime_error);
    ClefHeadConfig wider = c;
    wider.feedforward += 8;
    CHECK_THROWS(ClefHeadWeights::open(tiny_final() + "/joint_head.safetensors", wider), std::runtime_error);
}

SEPAL_TEST(clef, check_request_messages) {
    EncodedRequest ok;
    ok.token_ids = {1, 2, 3, 4};
    EncodedQuestion q;
    q.instruction = {0, 1};
    q.options = {{1, 2}, {2, 4}};
    ok.questions = {q};
    check_request(ok, 10);
    EncodedRequest bad = ok;
    bad.token_ids.clear();
    CHECK_THROWS(check_request(bad, 10), std::invalid_argument);
    bad = ok;
    bad.token_ids[2] = 10;
    CHECK_THROWS(check_request(bad, 10), std::invalid_argument);
    bad = ok;
    bad.questions.clear();
    CHECK_THROWS(check_request(bad, 10), std::invalid_argument);
    bad = ok;
    bad.questions[0].options.clear();
    CHECK_THROWS(check_request(bad, 10), std::invalid_argument);
    bad = ok;
    bad.questions[0].options[1] = {2, 5};
    CHECK_THROWS(check_request(bad, 10), std::invalid_argument);
    bad = ok;
    bad.questions[0].instruction = {2, 2};
    CHECK_THROWS(check_request(bad, 10), std::invalid_argument);
    bad = ok;
    bad.questions[0].type = static_cast<QuestionType>(3);
    CHECK_THROWS(check_request(bad, 10), std::invalid_argument);
}

SEPAL_TEST(clef, fixture_matches_pytorch_float64) {
    const TrainedModel m = TrainedModel::open(tiny_final(), tiny_base());
    const sepal::test::GoldenCase c = sepal::test::read_case(tiny_final() + "/head_case.safetensors");
    const RequestLogits got = clef_head_reference(m.head(), c.hidden, c.request, &embedding_callback, &m);
    const sepal::test::Parity parity = sepal::test::compare_logits(got, c.logits);
    std::cout << "  head vs PyTorch fp64 (fixture): max |dlogit| " << std::scientific << parity.max_dlogit << ", max |dp| " << parity.max_dp << "\n";
    CHECK_EQ(got.logits.size(), std::size_t{3});
    CHECK(parity.max_dlogit <= 1e-5);
    CHECK(parity.max_dp <= 1e-5);
    // a request with a hidden array of the wrong size is refused
    CHECK_THROWS(clef_head_reference(m.head(), std::span<const float>(c.hidden).first(c.hidden.size() - 1), c.request, &embedding_callback, &m),
                 std::invalid_argument);
}

SEPAL_TEST(clef_golden, golden_set) {
    const auto golden = env_value("SEPAL_GOLDEN_DIR");
    if (!golden || !sepal::test::real_model_available("clef golden")) {
        if (!golden) std::cout << "SKIPPED: real-model clef golden test; set SEPAL_GOLDEN_DIR (runs/runner-golden/full08b)\n";
        return;
    }
    const TrainedModel m = TrainedModel::open(*env_value("SEPAL_TRAINED_FINAL"), *env_value("SEPAL_TRAINED_BASE"));
    const json::Value manifest = json::parse(read_text(fs::path(*golden) / "manifest.json"));
    std::uint64_t max_tokens = 4096;
    if (const auto limit = env_value("SEPAL_GOLDEN_MAX_TOKENS")) max_tokens = std::stoull(*limit);
    double worst_logit = 0, worst_p = 0;
    std::size_t flips = 0, run = 0;
    for (const json::Value& entry : manifest.at("cases").items()) {
        if (entry.at("tokens").as_uint() > max_tokens) continue;
        const std::string file = (fs::path(*golden) / entry.at("file").as_string()).string();
        const sepal::test::GoldenCase c = sepal::test::read_case(file);
        const auto start = Clock::now();
        const RequestLogits got = clef_head_reference(m.head(), c.hidden, c.request, &embedding_callback, &m);
        const sepal::test::Parity parity = sepal::test::compare_logits(got, c.logits);
        std::cout << "  " << entry.at("name").as_string() << ": L=" << c.request.token_ids.size() << " max |dlogit| " << std::scientific << parity.max_dlogit
                  << " max |dp| " << parity.max_dp << " flips " << parity.flips << " (" << std::fixed << std::setprecision(2) << seconds_since(start) << " s)\n";
        worst_logit = std::fmax(worst_logit, parity.max_dlogit);
        worst_p = std::fmax(worst_p, parity.max_dp);
        flips += parity.flips;
        ++run;
    }
    std::cout << "  golden head: " << run << " case(s), max |dlogit| " << std::scientific << worst_logit << ", max |dp| " << worst_p << ", flips " << flips << "\n";
    CHECK(run > 0);
    CHECK(worst_logit <= 1e-4);
    CHECK(worst_p <= 1e-5);
    CHECK_EQ(flips, std::size_t{0});
}
