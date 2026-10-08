// A tiny test harness: self-registering cases and a few check macros. No third-party dependency.
#pragma once

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace sepal::test {

struct Case {
    const char* suite;
    const char* name;
    void (*fn)();
};

std::vector<Case>& registry();
void report(bool ok, const std::string& text, const char* file, int line);

struct Registrar {
    Registrar(const char* suite, const char* name, void (*fn)()) { registry().push_back({suite, name, fn}); }
};

template <typename A, typename B>
void check_eq(const A& a, const B& b, const char* a_text, const char* b_text, const char* file, int line) {
    const bool ok = a == b;
    std::ostringstream s;
    s << a_text << " == " << b_text;
    if (!ok) s << "  (" << a << " vs " << b << ")";
    report(ok, s.str(), file, line);
}

inline void check_near(double a, double b, double tol, const char* a_text, const char* b_text, const char* file,
                       int line) {
    const bool ok = std::fabs(a - b) <= tol;
    std::ostringstream s;
    s.precision(12);
    s << a_text << " ~= " << b_text;
    if (!ok) s << "  (" << a << " vs " << b << ", tolerance " << tol << ")";
    report(ok, s.str(), file, line);
}

} // namespace sepal::test

#define SEPAL_TEST(suite, name)                                                                        \
    static void sepal_test_##suite##_##name();                                                         \
    static const ::sepal::test::Registrar sepal_registrar_##suite##_##name(#suite, #name,              \
                                                                           &sepal_test_##suite##_##name); \
    static void sepal_test_##suite##_##name()

#define CHECK(cond) ::sepal::test::report(static_cast<bool>(cond), #cond, __FILE__, __LINE__)
#define CHECK_EQ(a, b) ::sepal::test::check_eq((a), (b), #a, #b, __FILE__, __LINE__)
#define CHECK_NEAR(a, b, tol) \
    ::sepal::test::check_near(static_cast<double>(a), static_cast<double>(b), (tol), #a, #b, __FILE__, __LINE__)
#define CHECK_THROWS(expr, exception_type)                                              \
    do {                                                                                \
        bool sepal_threw = false;                                                       \
        try {                                                                           \
            (void)(expr);                                                               \
        } catch (const exception_type&) {                                               \
            sepal_threw = true;                                                         \
        } catch (...) {                                                                 \
        }                                                                               \
        ::sepal::test::report(sepal_threw, #expr " throws " #exception_type, __FILE__, __LINE__); \
    } while (false)
