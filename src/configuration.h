#pragma once

#include <vector>
#include <array>
#include <unordered_map>

#include "particle.h"

enum class ConfigurationType {
	UNIFORM_SQUARE,
	UNIFORM_DISK,
	GAUSSIAN_CLOUD,
	PLUMMER_DISK
};

class Configuration
{
public:
	static std::vector<Particle> Generate(ConfigurationType type, auto&&... args)
	{
		if (type == ConfigurationType::UNIFORM_SQUARE)
			return GenerateUniformSquare(args...);
		else if (type == ConfigurationType::UNIFORM_DISK)
			return GenerateUniformDisk(args...);
		else if (type == ConfigurationType::GAUSSIAN_CLOUD)
			return GenerateGaussianCloud(args...);
		else
			return GeneratePlummerDisk(args...);
	}

	static ConfigurationType NameToType(const char* name);
	static const char* TypeToName(ConfigurationType type);
	static std::array<ConfigurationType, 4> GetTypes();

private:
	static std::vector<Particle> GenerateUniformSquare(int particle_count = 1000);
	static std::vector<Particle> GenerateUniformDisk(int particle_count = 1000);
	static std::vector<Particle> GenerateGaussianCloud(int particle_count = 1000, float variance = 0.1f);
	static std::vector<Particle> GeneratePlummerDisk(int particle_count = 1000, float plummer_radius = 0.1f);
};
