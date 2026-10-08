// A small JSON reader for the headers of the runner's own files. It parses the whole grammar of RFC 8259 except
// that numbers are held as float64 and \u escapes outside the Basic Multilingual Plane are not combined.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace sepal::json {

enum class Kind { Null, Bool, Number, String, Array, Object };

/// One JSON value. Every accessor throws std::runtime_error when the value is of another kind, and the message
/// says which kind was wanted, so a malformed header is reported and never misread.
class Value {
public:
    Kind kind() const { return kind_; }
    bool is_null() const { return kind_ == Kind::Null; }

    bool as_bool() const;
    double as_number() const;
    /// A number that is a non-negative whole value below 2^53.
    std::uint64_t as_uint() const;
    const std::string& as_string() const;

    /// The elements of an array, or the member values of an object in the order written.
    const std::vector<Value>& items() const;
    /// The member names of an object, parallel to items().
    const std::vector<std::string>& keys() const;

    /// Member of an object by name; nullptr when absent.
    const Value* find(std::string_view key) const;
    /// Member of an object by name; throws naming the key when absent.
    const Value& at(std::string_view key) const;
    /// Element of an array; throws when out of range.
    const Value& at(std::size_t index) const;

    /// An array of whole non-negative numbers.
    std::vector<std::uint64_t> as_uint_list() const;

private:
    friend class Parser;
    Kind kind_ = Kind::Null;
    bool bool_ = false;
    double number_ = 0.0;
    std::string string_;
    std::vector<Value> items_;
    std::vector<std::string> keys_;
};

/// Parses exactly one JSON value surrounded by optional whitespace. Throws std::runtime_error with the byte
/// offset of the first error.
Value parse(std::string_view text);

} // namespace sepal::json
