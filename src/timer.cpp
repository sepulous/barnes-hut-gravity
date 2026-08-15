#include <algorithm>

#include "timer.h"

std::vector<std::pair<size_t, Timer::Clock::time_point>> Timer::start_points;
std::unordered_map<size_t, Timer::Clock::duration> Timer::elapsed_times;

void Timer::Start(const std::string& label)
{
	size_t hash = std::hash<std::string>{}(label);
	auto time_point = Clock::now();
	start_points.emplace_back(hash, time_point);
}

void Timer::End()
{
	assert(start_points.size() > 0);

	auto [hash, start_point] = start_points.back();
	start_points.pop_back();

	if (elapsed_times.contains(hash))
		elapsed_times[hash] += Clock::now() - start_point;
	else
		elapsed_times[hash] = Clock::now() - start_point;
}
