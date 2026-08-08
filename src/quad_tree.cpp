#include "quad_tree.h"

#define TOP_LEFT 0
#define TOP_RIGHT 1
#define BOTTOM_RIGHT 2
#define BOTTOM_LEFT 3

size_t QuadTree::max_depth = 128;
size_t QuadTree::leaf_capacity = 8;

QuadTree::QuadTree(glm::dvec2 center, glm::dvec2 extents, size_t depth) : center_(center), extents_(extents), depth_(depth)
{
	total_mass_ = 0;
	center_of_mass_ = { 0, 0 };
	particles_.reserve(QuadTree::leaf_capacity);
}

void QuadTree::Add(Particle* point)
{
	if (HasChildren())
	{
		AddToChild(point);
	}
	else if (particles_.size() < QuadTree::leaf_capacity || depth_ >= QuadTree::max_depth)
	{
		particles_.push_back(point);
	}
	else
	{
		CreateChildren();

		AddToChild(point);

		for (auto particle : particles_)
			AddToChild(particle);
		particles_.clear();
    }
}

void QuadTree::AddToChild(Particle* point)
{
	glm::dvec2 pos = point->position;
	if (pos.x < center_.x && pos.y >= center_.y)
	{
		children_[TOP_LEFT].Add(point);
	}
	else if (pos.x >= center_.x && pos.y >= center_.y)
	{
		children_[TOP_RIGHT].Add(point);
	}
	else if (pos.x < center_.x && pos.y < center_.y)
	{
		children_[BOTTOM_LEFT].Add(point);
	}
	else
	{
		children_[BOTTOM_RIGHT].Add(point);
	}
}

void QuadTree::CreateChildren()
{
	children_.reserve(4);

	// Top left
	children_.emplace_back(
		glm::dvec2{
			center_.x - 0.5 * extents_.x,
			center_.y + 0.5 * extents_.y
		},
		glm::dvec2{
			0.5 * extents_.x,
			0.5 * extents_.y
		},
		depth_ + 1
	);

	// Top right
	children_.emplace_back(
		glm::dvec2{
			center_.x + 0.5 * extents_.x,
			center_.y + 0.5 * extents_.y
		},
		glm::dvec2{
			0.5 * extents_.x,
			0.5 * extents_.y
		},
		depth_ + 1
	);

	// Bottom right
	children_.emplace_back(
		glm::dvec2{
			center_.x + 0.5 * extents_.x,
			center_.y - 0.5 * extents_.y
		},
		glm::dvec2{
			0.5 * extents_.x,
			0.5 * extents_.y
		},
		depth_ + 1
	);

	// Bottom left
	children_.emplace_back(
		glm::dvec2{
			center_.x - 0.5 * extents_.x,
			center_.y - 0.5 * extents_.y
		},
		glm::dvec2{
			0.5 * extents_.x,
			0.5 * extents_.y
		},
		depth_ + 1
	);
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
