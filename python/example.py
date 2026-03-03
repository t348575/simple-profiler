import threading
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from simple_profiler import profiler, profile, profile_category, profile_scope


@profile
def do_work():
    for _ in range(10):
        time.sleep(0.001)


@profile_category("rendering")
def render_frame():
    with profile_scope("draw_meshes", "rendering"):
        time.sleep(0.005)
    with profile_scope("draw_ui", "rendering"):
        time.sleep(0.002)


def worker(name):
    with profile_scope(f"worker_{name}", "network"):
        time.sleep(0.01)
        do_work()


def main():
    profiler.begin_session("results.json")

    for i in range(10):
        with profile_scope("main_loop", "default"):
            time.sleep(0.001)
            do_work()

    render_frame()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    profiler.end_session()
    print("Trace written to results.json — open in https://perfetto.dev/")


if __name__ == "__main__":
    main()
