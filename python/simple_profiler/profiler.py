import functools
import json
import os
import threading
import time
from contextlib import contextmanager


class Profiler:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._active = False
                    cls._instance._events = []
                    cls._instance._events_lock = threading.Lock()
                    cls._instance._filepath = None
                    cls._instance._start_ts = None
        return cls._instance

    def begin_session(self, filepath="results.json"):
        if self._active:
            self.end_session()
        self._active = True
        self._filepath = filepath
        self._events = []
        self._start_ts = time.perf_counter_ns()

    def end_session(self):
        if not self._active:
            return
        self._active = False
        trace = {"traceEvents": self._events}
        with open(self._filepath, "w") as f:
            json.dump(trace, f)
        self._events = []

    def add_event(self, name, category, start_ns, duration_ns):
        if not self._active:
            return
        offset_us = (start_ns - self._start_ts) / 1000.0
        duration_us = duration_ns / 1000.0
        event = {
            "name": name,
            "cat": category,
            "ph": "X",
            "ts": offset_us,
            "dur": duration_us,
            "pid": os.getpid(),
            "tid": threading.get_ident(),
        }
        with self._events_lock:
            self._events.append(event)


profiler = Profiler()


@contextmanager
def profile_scope(name, category="default"):
    if not profiler._active:
        yield
        return
    start = time.perf_counter_ns()
    try:
        yield
    finally:
        duration = time.perf_counter_ns() - start
        profiler.add_event(name, category, start, duration)


def profile(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with profile_scope(func.__qualname__):
            return func(*args, **kwargs)
    return wrapper


def profile_category(category):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with profile_scope(func.__qualname__, category):
                return func(*args, **kwargs)
        return wrapper
    return decorator
