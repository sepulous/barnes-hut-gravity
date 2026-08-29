
#include <random>

#include <glm/glm.hpp>

#include "configuration.h"

static float RandRange(float min_inclusive, float max_inclusive)
{
	static std::mt19937_64 generator{ std::random_device{}() };
	std::uniform_real_distribution<float> dist(min_inclusive, max_inclusive);
	return dist(generator);
}

std::array<ConfigurationType, 4> GetConfigTypes()
{
	static std::array<ConfigurationType, 4> types {
		ConfigurationType::UNIFORM_SQUARE,
		ConfigurationType::UNIFORM_DISK,
		ConfigurationType::GAUSSIAN_CLOUD,
		ConfigurationType::PLUMMER_DISK
	};
	return types;
}

ConfigurationType ConfigNameToType(const char* name)
{
	if (name == "Uniform Square")
		return ConfigurationType::UNIFORM_SQUARE;
	else if (name == "Uniform Disk")
		return ConfigurationType::UNIFORM_DISK;
	else if (name == "Gaussian Cloud")
		return ConfigurationType::GAUSSIAN_CLOUD;
	else
		return ConfigurationType::PLUMMER_DISK;
}

const char* ConfigTypeToName(ConfigurationType type)
{
	if (type == ConfigurationType::UNIFORM_SQUARE)
		return "Uniform Square";
	else if (type == ConfigurationType::UNIFORM_DISK)
		return "Uniform Disk";
	else if (type == ConfigurationType::GAUSSIAN_CLOUD)
		return "Gaussian Cloud";
	else
		return "Plummer Disk";
}

std::vector<Particle> GenerateUniformSquare(int particle_count, float angular_speed)
{
	std::vector<Particle> particles;
	particles.reserve(particle_count);

	for (int i = 0; i < particle_count; i++)
	{
		glm::vec2 position = { RandRange(-1.0f, 1.0f), RandRange(-1.0f, 1.0f) };
		float r = glm::length(position);
		float theta = glm::atan(position.y / position.x);
		particles.push_back({
			.position = position,
			.velocity = r * angular_speed * glm::vec2{-glm::sin(theta), glm::cos(theta)},
			.acceleration = {0.0f, 0.0f},
			.mass = 1.0f
		});
	}

	return particles;
}

std::vector<Particle> GenerateUniformDisk(int particle_count, float angular_speed)
{
	std::vector<Particle> particles;
	particles.reserve(particle_count);

	for (int i = 0; i < particle_count; i++)
	{
		float u = RandRange(0.0f, 1.0f);
		float v = RandRange(0.0f, 1.0f);
		float r = 1.0f * glm::sqrt(u); // 1.0f = R
		float theta = 2.0f * 3.14159f * v;
		particles.push_back({
			.position = {
				r * glm::cos(theta),
				r * glm::sin(theta)
			},
			.velocity = r * angular_speed * glm::vec2{-glm::sin(theta), glm::cos(theta)},
			.acceleration = {0.0f, 0.0f},
			.mass = 1.0f
		});
	}

	return particles;
}

std::vector<Particle> GenerateGaussianCloud(int particle_count, float variance, float angular_speed)
{
	std::vector<Particle> particles;
	particles.reserve(particle_count);

	for (int i = 0; i < particle_count; i++)
	{
		float u = RandRange(0.0f, 1.0f);
		float v = RandRange(0.0f, 1.0f);
		float r = glm::sqrt(variance) * glm::sqrt(-2.0f * glm::log(u));
		float theta = 2.0f * 3.14159f * v;
		particles.push_back({
			.position = {
				r * glm::cos(theta),
				r * glm::sin(theta)
			},
			.velocity = r * angular_speed * glm::vec2{-glm::sin(theta), glm::cos(theta)},
			.acceleration = {0.0f, 0.0f},
			.mass = 1.0f
		});
	}

	return particles;
}

std::vector<Particle> GeneratePlummerDisk(int particle_count, float plummer_radius, float angular_speed)
{
	std::vector<Particle> particles;
	particles.reserve(particle_count);

	for (int i = 0; i < particle_count; i++)
	{
		float u = RandRange(0.0f, 1.0f);
		float v = RandRange(0.0f, 1.0f);
		float r = plummer_radius * glm::sqrt(glm::pow(1 - u, -2.0f / 3.0f) - 1.0f);
		float theta = 2.0f * 3.14159f * v;
		particles.push_back({
			.position = {
				r * glm::cos(theta),
				r * glm::sin(theta)
			},
			.velocity = r * angular_speed * glm::vec2{-glm::sin(theta), glm::cos(theta)},
			.acceleration = {0.0f, 0.0f},
			.mass = 1.0f
		});
	}

	return particles;
}
