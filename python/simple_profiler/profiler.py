import atexit
import functools
import json
import os
import signal
import tempfile
import threading
import time
from contextlib import contextmanager


_DISABLED = os.environ.get("SIMPLE_PROFILER_DISABLE", "").lower() in ("1", "true", "yes")


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
                    cls._instance._gpu_events = []  # deferred: (name, cat, wall_start_ns, cuda_start, cuda_end)
                    cls._instance._events_lock = threading.Lock()
                    cls._instance._filepath = None
                    cls._instance._start_ts = None
                    cls._instance._prev_sigint = None
                    cls._instance._prev_sigterm = None
                    cls._instance._merge_mode = False
                    cls._instance._session_count = 0
                    # GPU time reference: a CUDA event recorded + sync'd at session
                    # start, paired with the CPU wall-clock time at that moment.
                    # Used to convert CUDA elapsed_time() offsets into real wall-clock
                    # timestamps so that GPU events appear at their true GPU execution
                    # time rather than their CPU submission time.
                    cls._instance._gpu_ref_event = None
                    cls._instance._gpu_ref_wall_ns = None
                    atexit.register(cls._instance.end_session)
        return cls._instance

    def _sigint_handler(self, sig, frame):
        self.end_session()
        if callable(self._prev_sigint):
            self._prev_sigint(sig, frame)
        else:
            raise KeyboardInterrupt

    def _sigterm_handler(self, sig, frame):
        self.end_session()
        if callable(self._prev_sigterm):
            self._prev_sigterm(sig, frame)
        else:
            raise SystemExit(0)

    def begin_session(self, filepath="results.json", merge=False):
        if _DISABLED:
            return
        if merge:
            if self._session_count == 0:
                # First session in merge mode: full init
                self._merge_mode = True
                self._active = True
                self._filepath = filepath
                self._events = []
                self._gpu_events = []
                self._start_ts = time.perf_counter_ns()
                self._prev_sigint = signal.signal(signal.SIGINT, self._sigint_handler)
                self._prev_sigterm = signal.signal(signal.SIGTERM, self._sigterm_handler)
            self._session_count += 1
            return
        # Non-merge mode: original behaviour
        if self._active:
            self.end_session()
        self._merge_mode = False
        self._session_count = 1
        self._active = True
        self._filepath = filepath
        self._events = []
        self._gpu_events = []
        self._start_ts = time.perf_counter_ns()
        self._prev_sigint = signal.signal(signal.SIGINT, self._sigint_handler)
        self._prev_sigterm = signal.signal(signal.SIGTERM, self._sigterm_handler)
        # Record a GPU reference event and pair it with the CPU wall-clock time
        # after synchronising, so GPU-event start times can be anchored to the
        # real GPU execution timeline instead of the CPU submission timeline.
        self._gpu_ref_event = None
        self._gpu_ref_wall_ns = None
        try:
            import torch
            if torch.cuda.is_available():
                ref = torch.cuda.Event(enable_timing=True)
                ref.record()
                torch.cuda.synchronize()
                self._gpu_ref_wall_ns = time.perf_counter_ns()
                self._gpu_ref_event = ref
        except Exception as e:
            print(f"[profiler] GPU reference event setup failed: {e}")
        print("starting session")

    def end_session(self):
        if not self._active:
            return
        if self._merge_mode:
            self._session_count -= 1
            if self._session_count > 0:
                return
        import traceback
        is_main = threading.current_thread() is threading.main_thread()
        print(f"ending session for {self._filepath} (main_thread={is_main}, tid={threading.get_ident()})")
        # Resolve deferred GPU events before deactivating, so add_event's
        # _active guard doesn't drop them.
        # Deduplicate by CUDA event identity in case the same event pair was
        # registered more than once (e.g. due to concurrent access).
        seen_cuda_events: set[int] = set()
        for name, cat, wall_start_ns, cuda_start, cuda_end, args, tids in self._gpu_events:
            event_id = id(cuda_start)
            if event_id in seen_cuda_events:
                continue
            seen_cuda_events.add(event_id)
            try:
                cuda_end.synchronize()
                duration_ns = int(cuda_start.elapsed_time(cuda_end) * 1e6)  # ms -> ns
                # Anchor the start time to the true GPU execution time.
                # ref_event.elapsed_time(cuda_start) gives the GPU-clock delta
                # (in ms) from the reference point to when cuda_start fired on
                # the GPU.  Adding that to the reference wall-clock time gives
                # the real wall-clock instant the GPU began this scope, fixing
                # the async-scheduling overlap artefact.
                if self._gpu_ref_event is not None:
                    try:
                        start_offset_ns = int(
                            self._gpu_ref_event.elapsed_time(cuda_start) * 1e6
                        )
                        start_ns = self._gpu_ref_wall_ns + start_offset_ns
                    except Exception:
                        start_ns = wall_start_ns  # fallback to CPU submission time
                else:
                    start_ns = wall_start_ns
                if tids:
                    for tid, tid_args in tids:
                        self.add_event(name, cat, start_ns, duration_ns, tid=tid, args=tid_args)
                else:
                    self.add_event(name, cat, start_ns, duration_ns, args=args)
            except Exception as e:
                print(f"[profiler] GPU event resolution failed: {e}")
        self._gpu_events = []
        self._active = False
        try:
            if self._prev_sigint is not None:
                signal.signal(signal.SIGINT, self._prev_sigint)
                self._prev_sigint = None
            if self._prev_sigterm is not None:
                signal.signal(signal.SIGTERM, self._prev_sigterm)
                self._prev_sigterm = None
        except Exception as e:
            print(f"[profiler] signal restore failed: {e}")
            traceback.print_exc()
        print(f"[profiler] writing {len(self._events)} events to {self._filepath}")
        trace = {"traceEvents": self._events}
        # Serialize to a string first, then write atomically with signals blocked
        # so an arriving SIGINT/SIGTERM can't truncate the file mid-write.
        try:
            data = json.dumps(trace)
        except Exception as e:
            print(f"[profiler] json.dumps failed: {e}")
            traceback.print_exc()
            return
        dirpath = os.path.dirname(os.path.abspath(self._filepath))
        try:
            fd, tmp_path = tempfile.mkstemp(dir=dirpath, suffix=".tmp")
        except Exception as e:
            print(f"[profiler] tempfile.mkstemp failed: {e}")
            traceback.print_exc()
            return
        try:
            blocked = signal.pthread_sigmask(
                signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM}
            )
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(data)
                os.replace(tmp_path, self._filepath)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, blocked)
        except Exception as e:
            print(f"[profiler] write failed: {e}")
            traceback.print_exc()
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            return
        self._events = []
        self._merge_mode = False
        self._session_count = 0
        print(f"[profiler] wrote {self._filepath}")

    def add_gpu_event(self, name, category, wall_start_ns, cuda_start, cuda_end,
                      args=None, tids=None):
        if not self._active:
            return
        with self._events_lock:
            self._gpu_events.append((name, category, wall_start_ns, cuda_start, cuda_end, args, tids))

    def add_event(self, name, category, start_ns, duration_ns, tid=None, args=None):
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
            "tid": tid if tid is not None else threading.get_ident(),
        }
        if args is not None:
            event["args"] = args
        with self._events_lock:
            self._events.append(event)


profiler = Profiler()


@contextmanager
def profile_gpu_scope(name, category="default", args=None, tids=None):
    if not profiler._active:
        yield
        return
    import torch
    cuda_start = torch.cuda.Event(enable_timing=True)
    cuda_end = torch.cuda.Event(enable_timing=True)
    wall_start_ns = time.perf_counter_ns()
    cuda_start.record()
    try:
        yield
    finally:
        cuda_end.record()
        profiler.add_gpu_event(name, category, wall_start_ns, cuda_start, cuda_end,
                               args=args, tids=tids)


def profile_gpu(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with profile_gpu_scope(func.__qualname__):
            return func(*args, **kwargs)
    return wrapper


@contextmanager
def profile_scope(name, category="default", args=None):
    if not profiler._active:
        yield
        return
    start = time.perf_counter_ns()
    try:
        yield
    finally:
        duration = time.perf_counter_ns() - start
        profiler.add_event(name, category, start, duration, args=args)


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
