// The model loader on the tiny checkpoint in tests/fixtures/tiny_checkpoint, which tools/make_tiny_checkpoint.py
// writes: config.json in the transformers 5 layout, two bfloat16 shards with an index under the prefix
// `model.language_model.`, three tensors to ignore, a decision readout, and the float64 answers of the NumPy
// reference engine for the same bfloat16 values.

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "sepal/json.hpp"
#include "sepal/loader.hpp"
#include "sepal/ref/executor.hpp"
#include "test.hpp"

#ifndef SEPAL_FIXTURE_DIR
#error "SEPAL_FIXTURE_DIR must be defined by the build"
#endif

using namespace sepal;

namespace {

std::string fixture(const std::string& name) { return std::string(SEPAL_FIXTURE_DIR) + "/tiny_checkpoint/" + name; }

std::string read_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::string replaced(std::string text, const std::string& from, const std::string& to) {
    const std::size_t at = text.find(from);
    if (at == std::string::npos) throw std::logic_error("the fixture config has no '" + from + "'");
    return text.replace(at, from.size(), to);
}

// The `n`th (1-based) occurrence of `from` replaced.
std::string replaced_nth(std::string text, const std::string& from, int n, const std::string& to) {
    std::size_t at = std::string::npos;
    for (int i = 0; i < n; ++i) {
        at = text.find(from, at == std::string::npos ? 0 : at + from.size());
        if (at == std::string::npos) throw std::logic_error("the fixture config has too few '" + from + "'");
    }
    return text.replace(at, from.size(), to);
}

// Whether reading `config` fails with std::invalid_argument naming `needle`.
bool config_refused(const std::string& config, const std::string& needle) {
    try {
        (void)read_model_config(config, "test");
    } catch (const std::invalid_argument& error) {
        return std::string(error.what()).find(needle) != std::string::npos;
    }
    return false;
}

std::vector<double> numbers(const json::Value& list) {
    std::vector<double> out;
    for (const json::Value& v : list.items()) out.push_back(v.as_number());
    return out;
}

void put_u64(std::vector<unsigned char>& out, std::uint64_t n) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<unsigned char>(n >> (8 * i)));
}

// A safetensors file of F32 tensors of shape [1], named as given.
void write_scalars(const std::filesystem::path& path, const std::vector<std::string>& names) {
    std::string header = "{";
    for (std::size_t i = 0; i < names.size(); ++i) {
        header += (i == 0 ? "" : ",") + std::string("\"") + names[i] + "\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[" +
                  std::to_string(4 * i) + "," + std::to_string(4 * i + 4) + "]}";
    }
    header += "}";
    header.append((8 - header.size() % 8) % 8, ' ');
    std::vector<unsigned char> bytes;
    put_u64(bytes, header.size());
    bytes.insert(bytes.end(), header.begin(), header.end());
    bytes.insert(bytes.end(), 4 * names.size(), 0);
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

} // namespace

SEPAL_TEST(loader, config_is_read_from_the_multimodal_layout) {
    const ConfigReading reading = read_model_config(read_text(fixture("config.json")), "tiny-checkpoint");
    const ModelConfig want = tiny_test_config();
    const ModelConfig& got = reading.config;
    CHECK_EQ(got.name, std::string("tiny-checkpoint"));
    CHECK_EQ(got.num_layers, want.num_layers);
    CHECK_EQ(got.full_attention_interval, want.full_attention_interval);
    CHECK_EQ(got.hidden_size, want.hidden_size);
    CHECK_EQ(got.intermediate_size, want.intermediate_size);
    CHECK_EQ(got.attn_heads, want.attn_heads);
    CHECK_EQ(got.attn_kv_heads, want.attn_kv_heads);
    CHECK_EQ(got.attn_head_dim, want.attn_head_dim);
    CHECK_EQ(got.dn_key_heads, want.dn_key_heads);
    CHECK_EQ(got.dn_value_heads, want.dn_value_heads);
    CHECK_EQ(got.dn_key_head_dim, want.dn_key_head_dim);
    CHECK_EQ(got.dn_value_head_dim, want.dn_value_head_dim);
    CHECK_EQ(got.conv_kernel, want.conv_kernel);
    CHECK_EQ(got.vocab_size, want.vocab_size);
    CHECK_EQ(got.rms_norm_eps, want.rms_norm_eps);
    CHECK_EQ(got.rope_theta, want.rope_theta);
    CHECK_EQ(got.partial_rotary_factor, want.partial_rotary_factor);
    CHECK_EQ(got.lm_head_tied, want.lm_head_tied);
    CHECK_EQ(got.pointer_dim, default_pointer_dim);
    // A tiny geometry matches no preset: the closest one is reported with a difference per field.
    CHECK(!reading.preset.empty());
    CHECK(reading.differences.size() >= 4);
    bool names_hidden = false;
    for (const std::string& line : reading.differences) names_hidden = names_hidden || line.rfind("hidden_size: checkpoint 24", 0) == 0;
    CHECK(names_hidden);
}

SEPAL_TEST(loader, a_real_geometry_matches_its_preset) {
    // The 0.8B geometry as a text-only config with the top-level layout of older files.
    const std::string flat = R"({"model_type":"qwen3_5_text","num_hidden_layers":24,"full_attention_interval":4,"hidden_size":1024,
        "intermediate_size":3584,"num_attention_heads":8,"num_key_value_heads":2,"head_dim":256,"linear_num_key_heads":16,
        "linear_num_value_heads":16,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
        "vocab_size":248320,"rms_norm_eps":1e-06,"rope_theta":10000000,"partial_rotary_factor":0.25,"tie_word_embeddings":true})";
    const ConfigReading reading = read_model_config(flat, "x");
    CHECK_EQ(reading.preset, std::string("0.8b"));
    CHECK(reading.differences.empty());
    // One changed number is reported by name.
    const ConfigReading other = read_model_config(replaced(flat, "\"vocab_size\":248320", "\"vocab_size\":151936"), "x");
    CHECK_EQ(other.preset, std::string("0.8b"));
    CHECK_EQ(other.differences.size(), std::size_t(1));
    CHECK_EQ(other.differences[0], std::string("vocab_size: checkpoint 151936, preset 248320"));
}

SEPAL_TEST(loader, unsupported_configs_are_refused_with_the_field_named) {
    const std::string good = read_text(fixture("config.json"));
    CHECK(config_refused(replaced(good, "\"attn_output_gate\": true", "\"attn_output_gate\": false"), "attn_output_gate"));
    CHECK(config_refused(replaced(good, "\"attention_bias\": false", "\"attention_bias\": true"), "attention_bias"));
    CHECK(config_refused(replaced(good, "\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\""), "hidden_act"));
    CHECK(config_refused(replaced(good, "\"mlp_only_layers\": []", "\"mlp_only_layers\": [0]"), "mlp_only_layers"));
    CHECK(config_refused(replaced(good, "\"qwen3_5_text\"", "\"llama\""), "model_type"));
    CHECK(config_refused(replaced(good, "\"rope_type\": \"default\"", "\"rope_type\": \"yarn\""), "rope_type"));
    CHECK(config_refused(replaced(good, "\"hidden_size\"", "\"hidden_sizes\""), "hidden_size"));
    CHECK(config_refused(replaced(good, "\"full_attention_interval\": 4", "\"full_attention_interval\": 2"), "disagrees with layer_types"));
    CHECK(config_refused(replaced(good, "\"vocab_size\": 64", "\"vocab_size\": -3"), "vocab_size"));
    CHECK(config_refused(replaced(good, "\"num_hidden_layers\": 8", "\"num_hidden_layers\": 7"), "layer_types lists 8 layers"));
    // layer_types that break the fixed interval: without a stated interval, the sixth layer is made full attention.
    const std::string unstated = replaced(good, "\"full_attention_interval\": 4,", "");
    CHECK(config_refused(replaced_nth(unstated, "\"linear_attention\"", 5, "\"full_attention\""), "fixed full-attention interval"));
    CHECK_EQ(read_model_config(unstated, "t").config.full_attention_interval, std::uint64_t(4)); // derived from layer_types
    CHECK(config_refused("[1]", "top level"));
    CHECK(config_refused("{", "not valid JSON"));
    // An invalid geometry (a rotary fraction that does not give a whole number of dimensions) fails validation.
    CHECK_THROWS(read_model_config(replaced(good, "\"partial_rotary_factor\": 0.25", "\"partial_rotary_factor\": 0.3"), "t"), std::invalid_argument);
}

SEPAL_TEST(loader, tiny_checkpoint_is_found_mapped_and_sorted) {
    const LoadedModel model = LoadedModel::open(std::string(SEPAL_FIXTURE_DIR) + "/tiny_checkpoint");
    CHECK_EQ(model.prefix(), std::string("model.language_model."));
    CHECK_EQ(model.checkpoint().files().size(), std::size_t(2));

    // Every tensor the geometry implies except the pointer head, which a checkpoint does not have.
    const NamedShapes expected = weight_shapes(model.config());
    CHECK_EQ(model.raw().size(), expected.size() - head_shapes(model.config()).size());
    for (const auto& [name, shape] : expected) {
        if (name.starts_with("head.")) {
            CHECK(model.raw().find(name) == model.raw().end());
            continue;
        }
        const RawTensor& raw = model.raw_tensor(name);
        CHECK(raw.shape == shape);
        CHECK(raw.dtype == DType::BF16);
        CHECK_EQ(raw.bytes, raw.elements() * 2);
    }
    CHECK_THROWS(model.raw_tensor("no.such.tensor"), std::out_of_range);

    // The three tensors a prefill engine does not load are listed with a reason, not silently dropped.
    CHECK_EQ(model.ignored().size(), std::size_t(3));
    std::vector<std::string> reasons;
    for (const IgnoredTensor& tensor : model.ignored()) reasons.push_back(tensor.name + ": " + tensor.reason);
    bool vision = false, lm_head = false, mtp = false;
    for (const std::string& line : reasons) {
        vision = vision || (line.starts_with("model.visual.") && line.find("vision tower") != std::string::npos);
        lm_head = lm_head || (line.starts_with("lm_head.weight") && line.find("vocabulary projection") != std::string::npos);
        mtp = mtp || (line.starts_with("mtp.") && line.find("multi-token-prediction") != std::string::npos);
    }
    CHECK(vision);
    CHECK(lm_head);
    CHECK(mtp);
    CHECK(model.raw_bytes() > 100000);
}

SEPAL_TEST(loader, weights_are_the_stored_values_widened) {
    const LoadedModel model = LoadedModel::open(std::string(SEPAL_FIXTURE_DIR) + "/tiny_checkpoint");
    const Weights<float> f = model.weights<float>();
    const Weights<double> d = model.weights<double>();
    CHECK_EQ(f.size(), weight_shapes(model.config()).size());
    for (const std::string name : {"embed_tokens.weight", "layers.1.linear_attn.conv1d.weight", "layers.3.self_attn.q_proj.weight",
                                   "norm.weight"}) {
        const RawTensor& raw = model.raw_tensor(name);
        const std::uint16_t* bits = reinterpret_cast<const std::uint16_t*>(raw.data);
        bool same = f[name].size() == raw.elements() && d[name].size() == raw.elements();
        for (std::size_t i = 0; same && i < raw.elements(); ++i) {
            same = f[name][i] == bf16_to_float(bits[i]) && d[name][i] == static_cast<double>(bf16_to_float(bits[i]));
        }
        CHECK(same);
    }
    // The pointer head is zero.
    CHECK_EQ(f["head.q.weight"].size(), std::size_t(model.config().pointer_dim * model.config().hidden_size));
    double head_sum = 0.0;
    for (const float v : f["head.k.weight"].span()) head_sum += std::fabs(static_cast<double>(v));
    CHECK_EQ(head_sum, 0.0);
}

SEPAL_TEST(loader, forward_pass_and_readout_match_the_numpy_engine) {
    const LoadedModel model = LoadedModel::open(std::string(SEPAL_FIXTURE_DIR) + "/tiny_checkpoint");
    const json::Value expected = json::parse(read_text(fixture("expected.json")));
    std::vector<std::uint32_t> tokens;
    for (const json::Value& t : expected.at("token_ids").items()) tokens.push_back(static_cast<std::uint32_t>(t.as_uint()));
    const std::vector<double> want_hidden = numbers(expected.at("hidden"));

    const auto gemm = make_reference_gemm();
    const Weights<double> weights = model.weights<double>();
    ref::Row<double> row;
    row.token_ids = tokens;
    row.readout_offsets = {tokens.size() - 1};
    const std::vector<ref::Row<double>> rows = {row};
    const auto result = ref::run_rows<double>(*gemm, model.config(), weights, rows);
    const auto hidden = result[0].hidden.mat().row(0);
    CHECK_EQ(hidden.size(), want_hidden.size());
    double worst = 0.0;
    for (std::size_t i = 0; i < hidden.size(); ++i) worst = std::max(worst, std::fabs(hidden[i] - want_hidden[i]));
    CHECK(worst < 1e-9);

    // float32 stays close to the float64 answer.
    const Weights<float> weights32 = model.weights<float>();
    ref::Row<float> row32;
    row32.token_ids = tokens;
    row32.readout_offsets = {tokens.size() - 1};
    const std::vector<ref::Row<float>> rows32 = {row32};
    const auto result32 = ref::run_rows<float>(*gemm, model.config(), weights32, rows32);
    double worst32 = 0.0;
    for (std::size_t i = 0; i < hidden.size(); ++i) worst32 = std::max(worst32, std::fabs(static_cast<double>(result32[0].hidden.mat().row(0)[i]) - want_hidden[i]));
    CHECK(worst32 < 1e-4);

    // The readout: bfloat16 rows, temperature from decision_config.json, float64 arithmetic.
    const DecisionReadout readout = DecisionReadout::open_directory(std::string(SEPAL_FIXTURE_DIR) + "/tiny_checkpoint");
    CHECK_EQ(readout.rows(), std::size_t(6));
    CHECK_EQ(readout.hidden_size(), std::size_t(24));
    CHECK_EQ(readout.temperature(), expected.at("temperature").as_number());
    for (const char* key : {"probabilities_3", "probabilities_6"}) {
        const std::vector<double> want = numbers(expected.at(key));
        const std::vector<double> got = readout.probabilities(std::span<const double>(hidden.data(), hidden.size()), want.size());
        double diff = 0.0, sum = 0.0;
        for (std::size_t i = 0; i < want.size(); ++i) {
            diff = std::max(diff, std::fabs(got[i] - want[i]));
            sum += got[i];
        }
        CHECK_EQ(got.size(), want.size());
        CHECK(diff < 1e-12);
        CHECK_NEAR(sum, 1.0, 1e-12);
    }
    const std::vector<double> from_f32 = readout.probabilities(result32[0].hidden.mat().row(0), 3);
    CHECK_NEAR(from_f32[0], numbers(expected.at("probabilities_3"))[0], 1e-4);
    CHECK_THROWS(readout.probabilities(std::span<const double>(hidden.data(), hidden.size()), 0), std::invalid_argument);
    CHECK_THROWS(readout.probabilities(std::span<const double>(hidden.data(), hidden.size()), 7), std::invalid_argument);
    CHECK_THROWS(readout.probabilities(std::span<const double>(hidden.data(), 5), 3), std::invalid_argument);
    CHECK_THROWS(DecisionReadout::open(fixture("readout.safetensors"), 0.0), std::invalid_argument);
}

SEPAL_TEST(loader, checkpoints_that_do_not_fit_are_refused) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "sepal_test_loader_bad";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const auto write_config = [&](const std::string& text) {
        std::ofstream(dir / "config.json", std::ios::binary) << text;
    };
    const std::string good = read_text(fixture("config.json"));

    CHECK_THROWS(LoadedModel::open(dir.string()), std::runtime_error); // no config.json
    write_config(good);
    CHECK_THROWS(LoadedModel::open(dir.string()), SafetensorsError); // no tensors at all

    write_scalars(dir / "model.safetensors", {"foo.bar"});
    CHECK_THROWS(LoadedModel::open(dir.string()), std::runtime_error); // no embedding under any prefix

    // The embedding is there, but its shape is a scalar's and there are tensors nobody knows.
    write_scalars(dir / "model.safetensors", {"embed_tokens.weight", "surprise.weight"});
    bool unexpected = false;
    try {
        (void)LoadedModel::open(dir.string());
    } catch (const std::runtime_error& error) {
        unexpected = std::string(error.what()).find("surprise.weight") != std::string::npos;
    }
    CHECK(unexpected);

    // Known names, wrong shapes: the strict check of the engine's weights names the first problem.
    write_scalars(dir / "model.safetensors", {"embed_tokens.weight"});
    bool misfit = false;
    try {
        (void)LoadedModel::open(dir.string());
    } catch (const std::runtime_error& error) {
        misfit = std::string(error.what()).find("does not fit") != std::string::npos;
    }
    CHECK(misfit);
    std::filesystem::remove_all(dir);
}
