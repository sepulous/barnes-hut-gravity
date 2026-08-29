#pragma once

#include <vector>
#include <array>

#include "particle.h"

enum class ConfigurationType
{
	UNIFORM_SQUARE,
	UNIFORM_DISK,
	GAUSSIAN_CLOUD,
	PLUMMER_DISK
};

struct ConfigurationSettings
{
	ConfigurationType type;
	int particle_count;
	float angular_speed;
	float variance;
	float plummer_radius;
};

std::vector<Particle> GenerateUniformSquare(int particle_count = 1000, float angular_speed = 0);
std::vector<Particle> GenerateUniformDisk(int particle_count = 1000, float angular_speed = 0);
std::vector<Particle> GenerateGaussianCloud(int particle_count = 1000, float variance = 0.1f, float angular_speed = 0);
std::vector<Particle> GeneratePlummerDisk(int particle_count = 1000, float plummer_radius = 0.1f, float angular_speed = 0);
ConfigurationType ConfigNameToType(const char* name);
const char* ConfigTypeToName(ConfigurationType type);
std::array<ConfigurationType, 4> GetConfigTypes();
