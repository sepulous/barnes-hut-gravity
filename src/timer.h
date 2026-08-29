#pragma once

#include <vector>
#include <unordered_map>
#include <chrono>
#include <cassert>

class Timer
{
public:
	using Clock = std::chrono::high_resolution_clock;

	static void Start(const std::string& label);
	static void End();

	template <typename D = Clock::duration>
	static long long GetElapsed(const std::string& label)
	{
		size_t hash = std::hash<std::string>{}(label);
		assert(elapsed_times.contains(hash));
		return std::chrono::duration_cast<D>(elapsed_times[hash]).count();
	}

private:
	static std::vector<std::pair<size_t, Clock::time_point>> start_points;
	static std::unordered_map<size_t, Clock::duration> elapsed_times;

private:
	Timer();
};
