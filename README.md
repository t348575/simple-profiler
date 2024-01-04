# Simple header-only profiler wrapper over the [perfetto](https://perfetto.dev/) SDK ![Visits](https://nkvnu62257.execute-api.ap-south-1.amazonaws.com/production?repo=simple-profiler)

Check [src/main.cc](src/main.cc) for an example usage

### Instructions
* Include the header [src/VisualProfiler.hpp](src/VisualProfiler.hpp)
* Link the perfetto SDK
* Add `PROFILER_STORAGE()` to your entrypoint file
* Use `PROFILER_INIT()` to initialize the trace
* Use `PROFILER_DEFINE_CATEGORIES` to set trace categories
* Use `PROFILE_FUNCTION`, `PROFILE_SCOPE` etc. to trace scopes
* Run your code and load the `results.data` file into perfetto ui