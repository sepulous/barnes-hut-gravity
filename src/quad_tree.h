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
	QuadTree(glm::dvec2 center, glm::dvec2 extents, size_t depth = 0);
	~QuadTree() = default;

	QuadTree(const QuadTree&) = delete;
	QuadTree& operator=(const QuadTree&) = delete;

	QuadTree(QuadTree&&) = default;
	QuadTree& operator=(QuadTree&&) = default;

	void Build(std::span<Particle> particles);
	void CalculateMass();
	size_t GetDepth() const;
	glm::dvec2 GetCenter() const;
	glm::dvec2 GetExtents() const;
	glm::dvec2 GetCenterOfMass() const;
	double GetTotalMass() const;
	bool HasChildren() const;
	std::array<uint32_t, 4>& GetChildren();
	std::span<Particle>& GetParticles();

	static void SetMaxDepth(size_t max_depth);
	static void SetLeafCapacity(size_t leaf_capacity);
	static std::vector<QuadTree>& GetPool();

private:
	std::span<Particle> particles_;
	std::array<uint32_t, 4> children_;
	glm::dvec2 center_;
	glm::dvec2 extents_;
	glm::dvec2 center_of_mass_;
	double total_mass_;
	size_t depth_;
	bool has_children_ = false;

	static size_t max_depth;
	static size_t leaf_capacity;

	static std::vector<QuadTree> pool;
};
