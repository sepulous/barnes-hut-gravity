#include "quad_tree.h"

#include <algorithm>

#include <iostream>

#define TOP_LEFT 0
#define TOP_RIGHT 1
#define BOTTOM_RIGHT 2
#define BOTTOM_LEFT 3

size_t QuadTree::max_depth = 8;
size_t QuadTree::leaf_capacity = 64;
std::vector<QuadTree> QuadTree::pool;

QuadTree::QuadTree(glm::dvec2 center, glm::dvec2 extents, size_t depth) : center_(center), extents_(extents), depth_(depth)
{
	total_mass_ = 0;
	center_of_mass_ = { 0, 0 };
}

void QuadTree::Build(std::span<Particle> particles)
{
	if (particles.size() <= QuadTree::leaf_capacity || depth_ >= QuadTree::max_depth)
	{
		particles_ = particles;
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

		size_t this_index = static_cast<size_t>(this - pool.data());
		size_t top_left_index = pool.size();
		size_t top_right_index = top_left_index + 1;
		size_t bottom_right_index = top_right_index + 1;
		size_t bottom_left_index = bottom_right_index + 1;

		auto center = center_;
		auto extents = extents_;
		auto depth = depth_;

		// Top left
		pool.emplace_back(
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
		pool[this_index].children_[TOP_LEFT] = top_left_index;

		// Top right
		pool.emplace_back(
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
		pool[this_index].children_[TOP_RIGHT] = top_right_index;

		// Bottom right
		pool.emplace_back(
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
		pool[this_index].children_[BOTTOM_RIGHT] = bottom_right_index;

		// Bottom left
		pool.emplace_back(
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
		pool[this_index].children_[BOTTOM_LEFT] = bottom_left_index;

		pool[this_index].has_children_ = true;

		pool[top_left_index].Build(std::span<Particle>(top_left, bottom_left));
		pool[top_right_index].Build(std::span<Particle>(top_right, bottom_right));
		pool[bottom_right_index].Build(std::span<Particle>(bottom_right, particles.end()));
		pool[bottom_left_index].Build(std::span<Particle>(bottom_left, right_half));
	}
}

void QuadTree::CalculateMass()
{
	// Assumes total_mass_ and center_of_mass_ are zero

	if (!HasChildren()) // Leaf node
	{
		for (auto& particle : particles_)
		{
			total_mass_ += particle.mass;
			center_of_mass_ += particle.position;
		}
	}
	else // Internal node
	{
		for (auto child_index : GetChildren())
		{
			auto& child = pool[child_index];
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

size_t QuadTree::GetDepth() const
{
	return depth_;
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

std::array<uint32_t, 4>& QuadTree::GetChildren()
{
	return children_;
}

bool QuadTree::HasChildren() const
{
	return has_children_;
}

std::span<Particle>& QuadTree::GetParticles()
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

std::vector<QuadTree>& QuadTree::GetPool()
{
	return pool;
}
