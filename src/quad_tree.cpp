#include "quad_tree.h"

#include <algorithm>

#include <iostream>

#define TOP_LEFT 0
#define TOP_RIGHT 1
#define BOTTOM_RIGHT 2
#define BOTTOM_LEFT 3

unsigned QuadTree::max_depth = 8;
unsigned QuadTree::leaf_capacity = 64;
std::vector<QuadTree> QuadTree::pool;

QuadTree::QuadTree(glm::vec2 center, float width, unsigned depth) : center_(center), width_(width), depth_(depth)
{
	total_mass_ = 0;
	center_of_mass_ = { 0, 0 };
	first_child_index_ = 0;
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
		unsigned first_child_index = pool.size();
		auto center = center_;
		auto width = width_;
		auto depth = depth_;

		// Top left
		pool.emplace_back(
			glm::dvec2{
				center.x - 0.25 * width,
				center.y + 0.25 * width
			},
			0.5 * width,
			depth + 1
		);
		pool[this_index].first_child_index_ = first_child_index;

		// Top right
		pool.emplace_back(
			glm::dvec2{
				center.x + 0.25 * width,
				center.y + 0.25 * width
			},
			0.5 * width,
			depth + 1
		);

		// Bottom right
		pool.emplace_back(
			glm::dvec2{
				center.x + 0.25 * width,
				center.y - 0.25 * width
			},
			0.5 * width,
			depth + 1
		);

		// Bottom left
		pool.emplace_back(
			glm::dvec2{
				center.x - 0.25 * width,
				center.y - 0.25 * width
			},
			0.5 * width,
			depth + 1
		);

		pool[first_child_index].Build(std::span<Particle>(top_left, bottom_left));
		pool[first_child_index + 1].Build(std::span<Particle>(top_right, bottom_right));
		pool[first_child_index + 2].Build(std::span<Particle>(bottom_right, particles.end()));
		pool[first_child_index + 3].Build(std::span<Particle>(bottom_left, right_half));
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
		for (auto child_index = first_child_index_; child_index < first_child_index_ + 4; child_index++)
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

glm::vec2 QuadTree::GetCenter() const
{
	return center_;
}

float QuadTree::GetWidth() const
{
	return width_;
}

glm::vec2 QuadTree::GetCenterOfMass() const
{
	return center_of_mass_;
}

float QuadTree::GetTotalMass() const
{
	return total_mass_;
}

unsigned QuadTree::GetFirstChildIndex() const
{
	return first_child_index_;
}

bool QuadTree::HasChildren() const
{
	return first_child_index_ != 0;
}

const std::span<Particle>& QuadTree::GetParticles() const
{
	return particles_;
}

void QuadTree::SetMaxDepth(unsigned max_depth)
{
	QuadTree::max_depth = max_depth;
}

void QuadTree::SetLeafCapacity(unsigned leaf_capacity)
{
	QuadTree::leaf_capacity = leaf_capacity;
}

std::vector<QuadTree>& QuadTree::GetPool()
{
	return pool;
}
