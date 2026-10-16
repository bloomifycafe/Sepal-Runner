// The CPU library's profiles (sepal_runner_cpu): the C++ reference engines. rt-cpu adds its fast profiles here.
#include "sepal/engine.hpp"

namespace sepal {

void register_backend_profiles() { register_ref_profiles(); }

} // namespace sepal
