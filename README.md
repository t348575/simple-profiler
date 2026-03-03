# Simple profiler wrapper over the [perfetto](https://perfetto.dev/) SDK ![Visits](https://lambda.348575.xyz/repo-view-counter?repo=simple-profiler)

Run your code and load the output file into [Perfetto UI](https://perfetto.dev/) to visualize traces.

---

## C++

Header-only wrapper around the Perfetto SDK. See [src/main.cc](src/main.cc) for a full example.

### Instructions
* Include the header [src/VisualProfiler.hpp](src/VisualProfiler.hpp)
* Link the Perfetto SDK
* Add `PROFILER_STORAGE()` to your entrypoint file
* Use `PROFILER_INIT()` to initialize the trace
* Use `PROFILER_DEFINE_CATEGORIES` to set trace categories
* Use `PROFILE_FUNCTION`, `PROFILE_SCOPE` etc. to trace scopes
* Run your code and load `results.data` into Perfetto UI

---

## Python

Zero-dependency Python profiler using decorators and context managers (does not use the perfetto sdk). See [python/example.py](python/example.py) for a full example.

### Instructions
* Run `pip install .` from [python/](python/) after activating a venv.
* Call `profiler.begin_session()` before profiling and `profiler.end_session()` after
* Use `@profile` or `@profile_category("cat")` to trace functions
* Use `with profile_scope("name", "cat"):` to trace code blocks
* Run your code and load `results.json` into Perfetto UI

### Example

```python
from simple_profiler import profiler, profile, profile_category, profile_scope

profiler.begin_session("results.json")

@profile
def my_function():
    with profile_scope("inner_work", "rendering"):
        do_stuff()

@profile_category("network")
def fetch_data():
    ...

my_function()
fetch_data()

profiler.end_session()
```
