// Runs every registered test case, or only the suite named on the command line.
// A suite name that matches no case is an error, so a misspelt CTest entry cannot pass silently.

#include <cstring>
#include <exception>
#include <iostream>

#include "test.hpp"

namespace sepal::test {

namespace {
int g_checks = 0;
int g_failures = 0;
} // namespace

std::vector<Case>& registry() {
    static std::vector<Case> cases;
    return cases;
}

void report(bool ok, const std::string& text, const char* file, int line) {
    ++g_checks;
    if (ok) return;
    ++g_failures;
    std::cerr << file << ":" << line << ": FAILED: " << text << "\n";
}

} // namespace sepal::test

int main(int argc, char** argv) {
    using namespace sepal::test;
    const char* filter = argc > 1 ? argv[1] : nullptr;

    int cases_run = 0;
    int cases_failed = 0;
    for (const Case& c : registry()) {
        if (filter != nullptr && std::strcmp(filter, c.suite) != 0) continue;
        ++cases_run;
        const int failures_before = g_failures;
        try {
            c.fn();
        } catch (const std::exception& e) {
            report(false, std::string("unexpected exception: ") + e.what(), c.name, 0);
        } catch (...) {
            report(false, "unexpected exception of unknown type", c.name, 0);
        }
        const bool failed = g_failures != failures_before;
        if (failed) ++cases_failed;
        std::cout << (failed ? "[FAIL] " : "[ ok ] ") << c.suite << "." << c.name << "\n";
    }

    if (cases_run == 0) {
        std::cerr << "no test case matches suite '" << (filter != nullptr ? filter : "") << "'\n";
        return 1;
    }
    std::cout << cases_run << " case(s), " << g_checks << " check(s), " << g_failures << " failed check(s) in "
              << cases_failed << " case(s)\n";
    return g_failures == 0 ? 0 : 1;
}
