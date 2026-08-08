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
	QuadTree(std::span<Particle> particles, glm::dvec2 center, glm::dvec2 extents, size_t depth);
	~QuadTree() = default;

	QuadTree(const QuadTree&) = delete;
	QuadTree& operator=(const QuadTree&) = delete;

	QuadTree(QuadTree&&) = default;
	QuadTree& operator=(QuadTree&&) = default;

	void CalculateMass();
	glm::dvec2 GetCenter() const;
	glm::dvec2 GetExtents() const;
	glm::dvec2 GetCenterOfMass() const;
	double GetTotalMass() const;
	bool HasChildren() const;
	std::vector<QuadTree>& GetChildren();
	std::vector<Particle*>& GetParticles();

	size_t GetDepth() { return depth_; }

	static void SetMaxDepth(size_t max_depth);
	static void SetLeafCapacity(size_t leaf_capacity);

private:
	std::vector<Particle*> particles_;
	std::vector<QuadTree> children_;
	glm::dvec2 center_;
	glm::dvec2 extents_;
	glm::dvec2 center_of_mass_;
	double total_mass_;
	size_t depth_;

	static size_t max_depth;
	static size_t leaf_capacity;
};
