// The trained model as Unsloth saved it (RUNNER.md 3.6): a base checkpoint directory (the local HF snapshot of
// Qwen/Qwen3.5-0.8B-Base @ dc7cdfe2...) and a `final/` directory with the LoRA adapter, the Clef head and the
// configs. Never reads a pickle; never touches the network.
//
// LoRA: PEFT keys `base_model.model.model.language_model.layers.N.<mixer>.<proj>.lora_{A,B}.weight`, float32,
// A [r, in], B [out, r], scale alpha / r (use_rslora false, dropout 0; refuse anything else: DoRA, rslora, bias,
// rank_pattern / alpha_pattern entries, modules_to_save, a target this engine does not know).
// Merge rule (Kev's, RUNNER.md 11): W' = float32(W_base) + (alpha / r) * (B @ A), computed in float32 (the product
// accumulated in float64 is fine), then rounded ONCE to the profile's storage type by the backend.
// The embedding and the final norm are not adapted. Every adapted module must be present in the base.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "sepal/clef_head.hpp"
#include "sepal/config.hpp"
#include "sepal/loader.hpp"
#include "sepal/tensor.hpp"
#include "sepal/weights.hpp"

namespace sepal {

struct LoraInfo {
    std::uint32_t rank = 0;
    double alpha = 0;
    double scale() const { return alpha / static_cast<double>(rank); }
    std::size_t modules = 0;         // adapted linear layers found (186 for the 0.8B: 18 x 8 + 6 x 7)
};

class TrainedModel {
public:
    /// `final_dir`: adapter_model.safetensors, adapter_config.json, joint_head.safetensors, joint_head_config.json,
    /// unsloth_decision_config.json, sepal.json. `base_dir`: config.json + model safetensors of the base (the
    /// adapter_config's `base_model_name_or_path` is a box path and is ignored; the caller names the base).
    /// Checks the base geometry against the head's hidden size and every adapter tensor against its base module.
    static TrainedModel open(const std::string& final_dir, const std::string& base_dir);

    const ModelConfig& config() const;
    const LoadedModel& base() const;
    const LoraInfo& lora() const;
    const ClefHeadConfig& head_config() const;
    /// Head parameters, float32 (bf16 upcast exactly; the three scalars are stored fp32).
    const ClefHeadWeights& head() const;

    /// The body with the LoRA merged in float32, one tensor per engine name (the same names and shapes as
    /// `LoadedModel::weights<float>()`, embedding included and unadapted). For T = double the merge is done in
    /// double (the fp64 reference). Built on demand; costs host memory (~3 GB at 0.8B for float).
    template <class T>
    Weights<T> merged_weights() const;

    /// One tensor by engine name, merged like `merged_weights` (the base tensor alone when the adapter has none for it).
    /// Throws std::out_of_range for a name the base does not have.
    template <class T>
    BasicTensor<T> merged_tensor(const std::string& name) const;

    /// Only the tensors whose engine name starts with `prefix + "."` (for example "layers.3" or "norm"), merged the same way and
    /// addressed relative to the prefix, without the geometry check. The CPU engine uses it to merge and pack one layer at a time,
    /// so the whole model is never held in float32 (at 4B that is 17 GB; one layer is under 0.4 GB).
    template <class T>
    Weights<T> merged_scope(const std::string& prefix) const;

    /// Row `token` of the (tied, unadapted) output embedding, float32, hidden_size values: the head's lexical vectors.
    void embedding_row(std::uint32_t token, float* out) const;

    /// `sepal.json` (the run's metadata, temperatures included) as text, for the caller; the engine applies none.
    const std::string& sepal_json() const;

    /// Wall-clock seconds `open` took (base mapping, head conversion, adapter validation; the merge is on demand).
    double load_seconds() const;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

} // namespace sepal
