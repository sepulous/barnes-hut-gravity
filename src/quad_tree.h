#pragma once

#include <vector>
#include <memory>
#include <array>
#include <cstdint>
#include <span>

#include <glm/glm.hpp>

#include "particle.h"

class QuadTree // 64 bytes (originally)
{
public:
	QuadTree() = delete;
	QuadTree(glm::vec2 center, float width, unsigned depth = 0);
	~QuadTree() = default;

	QuadTree(const QuadTree&) = delete;
	QuadTree& operator=(const QuadTree&) = delete;

	QuadTree(QuadTree&&) = default;
	QuadTree& operator=(QuadTree&&) = default;

	void Build(std::span<Particle> particles);
	void CalculateMass();
	size_t GetDepth() const;
	glm::vec2 GetCenter() const;
	float GetWidth() const;
	glm::vec2 GetCenterOfMass() const;
	float GetTotalMass() const;
	bool HasChildren() const;
	const std::array<uint32_t, 4>& GetChildren() const;
	const std::span<Particle>& GetParticles() const;

	static void SetMaxDepth(unsigned max_depth);
	static void SetLeafCapacity(unsigned leaf_capacity);
	static std::vector<QuadTree>& GetPool();

private:
	std::span<Particle> particles_;
	std::array<uint32_t, 4> children_{ 0,0,0,0 };
	glm::vec2 center_;
	glm::vec2 center_of_mass_;
	float width_;
	float total_mass_;
	unsigned depth_;

	static unsigned max_depth;
	static unsigned leaf_capacity;

	static std::vector<QuadTree> pool;
};
