#pragma once

#ifndef VISUAL_PROFILER_HPP
#define VISUAL_PROFILER_HPP

#include <fcntl.h>
#include <memory>

#ifdef PROFILING
#include <perfetto.h>
#endif

#ifndef PROFILING
#define PROFILER_DEFINE_CATEGORIES(...)
#define PROFILER_STORAGE()
#define PROFILER_INIT()

#define PROFILE_SCOPE(category, name)
#define PROFILE_FUNCTION_CATEGORY(category)
#define PROFILE_FUNCTION()
#else
#define PROFILER_DEFINE_CATEGORIES(...) PERFETTO_DEFINE_CATEGORIES(__VA_ARGS__);
#define PROFILER_STORAGE() PERFETTO_TRACK_EVENT_STATIC_STORAGE();

#define PROFILE_SCOPE(category, name) TRACE_EVENT(category, perfetto::DynamicString{name})
#define PROFILE_FUNCTION_CATEGORY(category)                                                                            \
	const char *ProfileFunction_##__LINE__ = __FUNCTION__;                                                             \
	PROFILE_SCOPE(category, ProfileFunction_##__LINE__)
#define PROFILE_FUNCTION() PROFILE_FUNCTION_CATEGORY("default");

#define PROFILER_INIT()                                                                                                \
	perfetto::TracingInitArgs args;                                                                                    \
	args.backends |= perfetto::kInProcessBackend;                                                                      \
	perfetto::Tracing::Initialize(args);                                                                               \
	perfetto::TrackEvent::Register();
#endif

class VisualProfiler {
public:
	static VisualProfiler &Instance() {
		static VisualProfiler instance;
		return instance;
	}

	~VisualProfiler() { endSession(); }

	void beginSession(const std::string &filepath = "results.data") {
#ifdef PROFILING
		if (m_activeSession) {
			endSession();
		}

		m_activeSession = true;
		perfetto::protos::gen::TrackEventConfig track_event_cfg;
		track_event_cfg.add_enabled_categories("*");

		perfetto::TraceConfig cfg;
		cfg.add_buffers()->set_size_kb(1024 * 128);
		cfg.set_file_write_period_ms(100);
		cfg.set_max_file_size_bytes(1024 * 1024 * 1024);
		auto *ds_cfg = cfg.add_data_sources()->mutable_config();
		ds_cfg->set_name("track_event");
		ds_cfg->set_track_event_config_raw(track_event_cfg.SerializeAsString());

		m_tracingSession = perfetto::Tracing::NewTrace();

		m_fd = open(filepath.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0664);
		m_tracingSession->Setup(cfg, m_fd);
		m_tracingSession->StartBlocking();
#endif
	}
	void endSession() {
#ifdef PROFILING
		m_tracingSession->StopBlocking();
		close(m_fd);
#endif
	}

private:
	bool m_activeSession = false;
	int m_fd;
#ifdef PROFILING
	std::unique_ptr<perfetto::TracingSession> m_tracingSession;
#endif
	VisualProfiler() {}
};

#endif
