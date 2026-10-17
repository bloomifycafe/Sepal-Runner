// The serving contract every backend (CPU, CUDA) implements for the trained Sepal decision model: Qwen3.5 base +
// merged LoRA + Clef JointSchemaHead (docs/RUNNER.md section 3).
//
// The caller (the Python front end, sepal_runner) renders and tokenises the one-prompt layout of RUNNER.md 3.2 and
// hands the engine token ids plus the spans the head reads. The engine returns RAW logits per question, one per
// option in encoded order. Temperatures, softmax, key mapping and the HTTP API are the caller's: they are the same
// Python code the oracle uses (sepal.train.unsloth_arm.UnslothScorer), so post-processing cannot drift.
//
// Spans are half-open token offsets into `token_ids` of the whole sequence (prefix + state + schema + suffix),
// exactly the `question_span` / `option_spans` of Clef's EncodedQuestion after the schema offset is added.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace sepal {

/// Clef's order (`QUESTION_TYPES` in joint_schema_model.py): the index into the head's `type_embedding`.
/// Not Unsloth's temperature order (choice, score, noul): see RUNNER.md 3.5.
enum class QuestionType : std::uint32_t { Noul = 0, Choice = 1, Score = 2 };

struct TokenSpan {
    std::uint32_t begin = 0;
    std::uint32_t end = 0; // exclusive; end > begin
};

struct EncodedQuestion {
    QuestionType type = QuestionType::Noul;
    TokenSpan instruction;            // the instruction text alone
    std::vector<TokenSpan> options;   // each option's JSON alone, in encoded order; at least one
};

struct EncodedRequest {
    std::vector<std::uint32_t> token_ids;    // the whole sequence; the head's global vector is the LAST token
    std::vector<EncodedQuestion> questions;  // at least one, in prompt order
};

/// Raw head logits: logits[q][k] for question q and option k (encoded order). No temperature applied.
struct RequestLogits {
    std::vector<std::vector<float>> logits;
};

/// Where the time of one `score` call went (filled when the backend can; zeros otherwise). Wall-clock milliseconds.
struct ScoreTimes {
    double trunk_ms = 0;
    double head_ms = 0;
    double total_ms = 0;
    std::uint64_t tokens = 0;         // tokens actually run through the trunk (after any prefix reuse)
    std::uint64_t tokens_reused = 0;  // tokens served from a prefix snapshot
};

/// A loaded trained model on one backend and precision profile.
class DecisionEngine {
public:
    virtual ~DecisionEngine() = default;

    /// Scores many independent requests; results are in request order. A request's logits must not depend on the
    /// other requests of the call beyond the profile's stated tolerance (RUNNER.md section 14). Throws
    /// std::invalid_argument for a malformed request (bad span, token id out of range, too long) before doing work.
    virtual std::vector<RequestLogits> score(std::span<const EncodedRequest> requests) = 0;

    /// Profile id reported with every answer, e.g. "gpu-f16", "cpu-f32", "cpu-bf16", "cpu-q8".
    virtual std::string profile() const = 0;

    /// The longest sequence one request may have.
    virtual std::uint64_t max_tokens() const = 0;

    /// Hidden size of the trunk, for sepal_info (0: not reported).
    virtual std::uint64_t hidden_size() const { return 0; }

    /// Times of the last `score` call.
    virtual ScoreTimes last_times() const { return {}; }
};

} // namespace sepal
