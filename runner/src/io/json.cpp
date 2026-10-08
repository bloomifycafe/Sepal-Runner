#include "sepal/json.hpp"

#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace sepal::json {

namespace {

[[noreturn]] void wrong_kind(const char* wanted) {
    throw std::runtime_error(std::string("json: the value is not ") + wanted);
}

constexpr std::size_t max_depth = 64;

} // namespace

bool Value::as_bool() const {
    if (kind_ != Kind::Bool) wrong_kind("a boolean");
    return bool_;
}

double Value::as_number() const {
    if (kind_ != Kind::Number) wrong_kind("a number");
    return number_;
}

std::uint64_t Value::as_uint() const {
    const double v = as_number();
    if (!(v >= 0.0) || v > 9007199254740992.0 || std::floor(v) != v) wrong_kind("a non-negative whole number");
    return static_cast<std::uint64_t>(v);
}

const std::string& Value::as_string() const {
    if (kind_ != Kind::String) wrong_kind("a string");
    return string_;
}

const std::vector<Value>& Value::items() const {
    if (kind_ != Kind::Array && kind_ != Kind::Object) wrong_kind("an array or an object");
    return items_;
}

const std::vector<std::string>& Value::keys() const {
    if (kind_ != Kind::Object) wrong_kind("an object");
    return keys_;
}

const Value* Value::find(std::string_view key) const {
    if (kind_ != Kind::Object) wrong_kind("an object");
    for (std::size_t i = 0; i < keys_.size(); ++i) {
        if (keys_[i] == key) return &items_[i];
    }
    return nullptr;
}

const Value& Value::at(std::string_view key) const {
    const Value* found = find(key);
    if (found == nullptr) throw std::runtime_error("json: the object has no member \"" + std::string(key) + "\"");
    return *found;
}

const Value& Value::at(std::size_t index) const {
    if (kind_ != Kind::Array) wrong_kind("an array");
    if (index >= items_.size()) throw std::runtime_error("json: array index out of range");
    return items_[index];
}

std::vector<std::uint64_t> Value::as_uint_list() const {
    if (kind_ != Kind::Array) wrong_kind("an array");
    std::vector<std::uint64_t> out;
    out.reserve(items_.size());
    for (const Value& item : items_) out.push_back(item.as_uint());
    return out;
}

class Parser {
public:
    explicit Parser(std::string_view text) : text_(text) {}

    Value document() {
        Value v = value(0);
        skip_whitespace();
        if (pos_ != text_.size()) fail("unexpected text after the value");
        return v;
    }

private:
    [[noreturn]] void fail(const char* what) const {
        throw std::runtime_error("json: " + std::string(what) + " at byte " + std::to_string(pos_));
    }

    void skip_whitespace() {
        while (pos_ < text_.size()) {
            const char c = text_[pos_];
            if (c != ' ' && c != '\t' && c != '\n' && c != '\r') break;
            ++pos_;
        }
    }

    char peek() const {
        if (pos_ >= text_.size()) fail("unexpected end of text");
        return text_[pos_];
    }

    bool consume(char c) {
        if (pos_ < text_.size() && text_[pos_] == c) {
            ++pos_;
            return true;
        }
        return false;
    }

    void expect_word(std::string_view word) {
        if (text_.substr(pos_, word.size()) != word) fail("unknown literal");
        pos_ += word.size();
    }

    Value value(std::size_t depth) {
        if (depth > max_depth) fail("nesting too deep");
        skip_whitespace();
        Value v;
        const char c = peek();
        if (c == '{') {
            ++pos_;
            v.kind_ = Kind::Object;
            skip_whitespace();
            if (consume('}')) return v;
            for (;;) {
                skip_whitespace();
                if (peek() != '"') fail("expected a member name");
                v.keys_.push_back(string());
                skip_whitespace();
                if (!consume(':')) fail("expected ':'");
                v.items_.push_back(value(depth + 1));
                skip_whitespace();
                if (consume(',')) continue;
                if (consume('}')) return v;
                fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            ++pos_;
            v.kind_ = Kind::Array;
            skip_whitespace();
            if (consume(']')) return v;
            for (;;) {
                v.items_.push_back(value(depth + 1));
                skip_whitespace();
                if (consume(',')) continue;
                if (consume(']')) return v;
                fail("expected ',' or ']'");
            }
        }
        if (c == '"') {
            v.kind_ = Kind::String;
            v.string_ = string();
            return v;
        }
        if (c == 't') {
            expect_word("true");
            v.kind_ = Kind::Bool;
            v.bool_ = true;
            return v;
        }
        if (c == 'f') {
            expect_word("false");
            v.kind_ = Kind::Bool;
            return v;
        }
        if (c == 'n') {
            expect_word("null");
            return v;
        }
        v.kind_ = Kind::Number;
        v.number_ = number();
        return v;
    }

    static void append_utf8(std::string& out, unsigned code) {
        if (code < 0x80u) {
            out.push_back(static_cast<char>(code));
        } else if (code < 0x800u) {
            out.push_back(static_cast<char>(0xC0u | (code >> 6)));
            out.push_back(static_cast<char>(0x80u | (code & 0x3Fu)));
        } else {
            out.push_back(static_cast<char>(0xE0u | (code >> 12)));
            out.push_back(static_cast<char>(0x80u | ((code >> 6) & 0x3Fu)));
            out.push_back(static_cast<char>(0x80u | (code & 0x3Fu)));
        }
    }

    std::string string() {
        ++pos_; // the opening quote
        std::string out;
        for (;;) {
            if (pos_ >= text_.size()) fail("unterminated string");
            const char c = text_[pos_++];
            if (c == '"') return out;
            if (static_cast<unsigned char>(c) < 0x20u) fail("control character in a string");
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            if (pos_ >= text_.size()) fail("unterminated escape");
            const char e = text_[pos_++];
            switch (e) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                if (text_.size() - pos_ < 4) fail("short \\u escape");
                unsigned code = 0;
                for (int i = 0; i < 4; ++i) {
                    const char h = text_[pos_++];
                    unsigned digit = 0;
                    if (h >= '0' && h <= '9') digit = static_cast<unsigned>(h - '0');
                    else if (h >= 'a' && h <= 'f') digit = static_cast<unsigned>(h - 'a') + 10u;
                    else if (h >= 'A' && h <= 'F') digit = static_cast<unsigned>(h - 'A') + 10u;
                    else fail("bad hex digit in a \\u escape");
                    code = code * 16u + digit;
                }
                append_utf8(out, code);
                break;
            }
            default: fail("unknown escape");
            }
        }
    }

    double number() {
        const std::size_t start = pos_;
        const auto digits = [this] {
            const std::size_t before = pos_;
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
            return pos_ != before;
        };
        consume('-');
        if (!digits()) fail("expected a value");
        if (consume('.') && !digits()) fail("expected digits after the decimal point");
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            ++pos_;
            if (!consume('+')) consume('-');
            if (!digits()) fail("expected digits in the exponent");
        }
        // strtod needs a terminated string; a number is short, so the copy is cheap.
        const std::string token(text_.substr(start, pos_ - start));
        return std::strtod(token.c_str(), nullptr);
    }

    std::string_view text_;
    std::size_t pos_ = 0;
};

Value parse(std::string_view text) { return Parser(text).document(); }

} // namespace sepal::json
