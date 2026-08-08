#include "quad_tree.h"

#include <algorithm>

#define TOP_LEFT 0
#define TOP_RIGHT 1
#define BOTTOM_RIGHT 2
#define BOTTOM_LEFT 3

size_t QuadTree::max_depth = 16;
size_t QuadTree::leaf_capacity = 64;

QuadTree::QuadTree(std::span<Particle> particles, glm::dvec2 center, glm::dvec2 extents, size_t depth) : center_(center), extents_(extents), depth_(depth)
{
	total_mass_ = 0;
	center_of_mass_ = { 0, 0 };

	if (particles.size() <= QuadTree::leaf_capacity || depth >= QuadTree::max_depth)
	{
		particles_.reserve(particles.size());
		for (auto& particle : particles)
			particles_.push_back(&particle);
	}
	else
	{
		//
		// Partition [TL|BL|TR|BR]
		//

		// Partition left/right
		auto right_half = std::partition(particles.begin(), particles.end(),
			[this](auto& particle) { return particle.position.x < center_.x; });

		// Partition left top/bottom
		auto bottom_left = std::partition(particles.begin(), right_half,
			[this](auto& particle) { return particle.position.y >= center_.y; });

		auto top_left = particles.begin();

		// Partition right top/bottom
		auto bottom_right = std::partition(right_half, particles.end(),
			[this](auto& particle) { return particle.position.y >= center_.y; });

		auto top_right = right_half;

		//
		// Create children
		//

		children_.reserve(4);

		// Top left
		children_.emplace_back(
			std::span<Particle>(top_left, bottom_left),
			glm::dvec2{
				center.x - 0.5 * extents.x,
				center.y + 0.5 * extents.y
			},
			glm::dvec2{
				0.5 * extents.x,
				0.5 * extents.y
			},
			depth + 1
		);

		// Top right
		children_.emplace_back(
			std::span<Particle>(top_right, bottom_right),
			glm::dvec2{
				center.x + 0.5 * extents.x,
				center.y + 0.5 * extents.y
			},
			glm::dvec2{
				0.5 * extents.x,
				0.5 * extents.y
			},
			depth + 1
		);

		// Bottom right
		children_.emplace_back(
			std::span<Particle>(bottom_right, particles.end()),
			glm::dvec2{
				center.x + 0.5 * extents.x,
				center.y - 0.5 * extents.y
			},
			glm::dvec2{
				0.5 * extents.x,
				0.5 * extents.y
			},
			depth + 1
		);

		// Bottom left
		children_.emplace_back(
			std::span<Particle>(bottom_left, right_half),
			glm::dvec2{
				center.x - 0.5 * extents.x,
				center.y - 0.5 * extents.y
			},
			glm::dvec2{
				0.5 * extents.x,
				0.5 * extents.y
			},
			depth + 1
		);
	}
}

void QuadTree::CalculateMass()
{
	// Assumes total_mass_ and center_of_mass_ are zero

	if (!HasChildren()) // Leaf node
	{
		for (auto particle : particles_)
		{
			total_mass_ += particle->mass;
			center_of_mass_ += particle->position;
		}
	}
	else // Internal node
	{
		for (auto& child : GetChildren())
		{
			child.CalculateMass();
			auto child_total_mass = child.GetTotalMass();
			auto child_center_of_mass = child.GetCenterOfMass();
			total_mass_ += child_total_mass;
			center_of_mass_ += child_total_mass * child_center_of_mass;
		}
	}

	if (total_mass_ > 0)
		center_of_mass_ /= total_mass_;
}

glm::dvec2 QuadTree::GetCenter() const
{
	return center_;
}

glm::dvec2 QuadTree::GetExtents() const
{
	return extents_;
}

glm::dvec2 QuadTree::GetCenterOfMass() const
{
	return center_of_mass_;
}

double QuadTree::GetTotalMass() const
{
	return total_mass_;
}

std::vector<QuadTree>& QuadTree::GetChildren()
{
	return children_;
}

bool QuadTree::HasChildren() const
{
	return children_.size() == 4;
}

std::vector<Particle*>& QuadTree::GetParticles()
{
	return particles_;
}

void QuadTree::SetMaxDepth(size_t max_depth)
{
	QuadTree::max_depth = max_depth;
}

void QuadTree::SetLeafCapacity(size_t leaf_capacity)
{
	QuadTree::leaf_capacity = leaf_capacity;
}
