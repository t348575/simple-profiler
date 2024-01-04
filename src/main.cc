#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include "VisualProfiler.hpp"

PROFILER_DEFINE_CATEGORIES(perfetto::Category("default"),
						   perfetto::Category("rendering").SetDescription("Events from the graphics subsystem"),
						   perfetto::Category("network").SetDescription("Network upload and download statistics"));

PROFILER_STORAGE();

void doWork() {
	PROFILE_FUNCTION();
	for (int i = 0; i < 10; i++) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
}

int main() {
	PROFILER_INIT();
	VisualProfiler::Instance().beginSession();
	for (int i = 0; i < 10; i++) {
		PROFILE_SCOPE("rendering", "asd");
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
		doWork();
	}
	return 0;
}