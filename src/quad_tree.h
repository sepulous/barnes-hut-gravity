#pragma once

#include <vector>
#include <memory>
#include <array>
#include <cstdint>
#include <span>

#include <glm/glm.hpp>

#include "particle.h"

class QuadTree
{
public:
	QuadTree() = delete;
	QuadTree(glm::vec2 center, glm::vec2 extents, size_t depth = 0);
	~QuadTree() = default;

	QuadTree(const QuadTree&) = delete;
	QuadTree& operator=(const QuadTree&) = delete;

	QuadTree(QuadTree&&) = default;
	QuadTree& operator=(QuadTree&&) = default;

	void Build(std::span<Particle> particles);
	void CalculateMass();
	size_t GetDepth() const;
	glm::vec2 GetCenter() const;
	glm::vec2 GetExtents() const;
	glm::vec2 GetCenterOfMass() const;
	float GetTotalMass() const;
	bool HasChildren() const;
	const std::array<uint32_t, 4>& GetChildren() const;
	const std::span<Particle>& GetParticles() const;

	static void SetMaxDepth(size_t max_depth);
	static void SetLeafCapacity(size_t leaf_capacity);
	static std::vector<QuadTree>& GetPool();

private:
	std::span<Particle> particles_;
	std::array<uint32_t, 4> children_;
	glm::vec2 center_;
	glm::vec2 extents_;
	glm::vec2 center_of_mass_;
	float total_mass_;
	size_t depth_;
	bool has_children_ = false;

	static size_t max_depth;
	static size_t leaf_capacity;

	static std::vector<QuadTree> pool;
};
