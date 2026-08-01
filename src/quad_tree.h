#pragma once

#include <vector>
#include <memory>
#include <array>

#include <glm/glm.hpp>

#include "particle.h"

class QuadTree
{
public:
	QuadTree();
	QuadTree(glm::dvec2, glm::dvec2);
	~QuadTree() = default;

	QuadTree(const QuadTree&) = delete;
	QuadTree& operator=(const QuadTree&) = delete;

	QuadTree(QuadTree&&) = default;
	QuadTree& operator=(QuadTree&&) = default;

	void Add(Particle*);
	void CalculateMass();
	glm::dvec2 GetCenter() const;
	glm::dvec2 GetExtents() const;
	glm::dvec2 GetCenterOfMass() const;
	double GetTotalMass() const;
	std::vector<QuadTree>& GetChildren();
	bool HasChildren() const;
	Particle* GetParticle() const;
	void Print(int level = 0);

private:
	void AddToChild(Particle*);
	void CreateChildren();

private:
	Particle* particle_ = nullptr;
	std::vector<QuadTree> children_;
	glm::dvec2 center_;
	glm::dvec2 extents_;
	glm::dvec2 center_of_mass_;
	double total_mass_;
};
