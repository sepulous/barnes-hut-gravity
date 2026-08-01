#include "quad_tree.h"

#define TOP_LEFT 0
#define TOP_RIGHT 1
#define BOTTOM_RIGHT 2
#define BOTTOM_LEFT 3

// Invariant: A node either has a point and no children, children and no point, or neither (i.e. it never has both a point and children)

QuadTree::QuadTree()
{
	center_ = { 0, 0 };
	extents_ = { 0, 0 };
	total_mass_ = 0;
	center_of_mass_ = { 0, 0 };
}

QuadTree::QuadTree(glm::dvec2 center, glm::dvec2 extents) : center_(center), extents_(extents)
{
	total_mass_ = 0;
	center_of_mass_ = { 0, 0 };
}

void QuadTree::Add(Particle* point)
{
	glm::dvec2 pos = point->position;

	if (pos.x < center_.x - extents_.x || pos.x > center_.x + extents_.x || pos.y < center_.y - extents_.y || pos.y > center_.y + extents_.y)
		return;

	if (!particle_)
	{
		if (HasChildren())
		{
			AddToChild(point);
		}
		else
		{
			particle_ = point;
		}
	}
	else
	{
		if (!HasChildren())
			CreateChildren();

		AddToChild(point);
		AddToChild(particle_);
		particle_ = nullptr;
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

	children_.emplace_back(
		glm::dvec2{
			center_.x - 0.5 * extents_.x,
			center_.y + 0.5 * extents_.y
		},
		glm::dvec2{
			0.5 * extents_.x,
			0.5 * extents_.y
		}
	);

	children_.emplace_back(
		glm::dvec2{
			center_.x + 0.5 * extents_.x,
			center_.y + 0.5 * extents_.y
		},
		glm::dvec2{
			0.5 * extents_.x,
			0.5 * extents_.y
		}
	);

	children_.emplace_back(
		glm::dvec2{
			center_.x - 0.5 * extents_.x,
			center_.y - 0.5 * extents_.y
		},
		glm::dvec2{
			0.5 * extents_.x,
			0.5 * extents_.y
		}
	);

	children_.emplace_back(
		glm::dvec2{
			center_.x + 0.5 * extents_.x,
			center_.y - 0.5 * extents_.y
		},
		glm::dvec2{
			0.5 * extents_.x,
			0.5 * extents_.y
		}
	);
}

void QuadTree::CalculateMass()
{
	if (particle_) // Leaf node w/ point
	{
		total_mass_ = particle_->mass;
		center_of_mass_.x = particle_->position.x;
		center_of_mass_.y = particle_->position.y;
	}
	else
	{
		total_mass_ = 0;
		center_of_mass_ = { 0, 0 };
		if (HasChildren()) // Internal node
		{
			for (auto& child : GetChildren())
			{
				child.CalculateMass();
				auto child_total_mass = child.GetTotalMass();
				auto child_center_of_mass = child.GetCenterOfMass();
				total_mass_ += child_total_mass;
				center_of_mass_.x += child_total_mass * child_center_of_mass.x;
				center_of_mass_.y += child_total_mass * child_center_of_mass.y;
			}

			if (total_mass_ > 0)
			{
				center_of_mass_.x /= total_mass_;
				center_of_mass_.y /= total_mass_;
			}
		}
		else
		{
			// Empty leaf node (no mass/center of mass)
		}
	}
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

Particle* QuadTree::GetParticle() const
{
	return particle_;
}

void QuadTree::Print(int level)
{
	for (int i = 0; i < level; i++)
		printf("  ");

	if (level == 0)
		printf("Root\n");
	else if (particle_)
		printf("Leaf Node (pos = (%f, %f), mass = %f)\n", particle_->position.x, particle_->position.y, particle_->mass);
	else if (!HasChildren())
		printf("Leaf Node (empty)\n");
	else
		printf("Internal Node\n");

	if (HasChildren())
	{
		for (auto& child : GetChildren())
		{
			child.Print(level + 1);
		}
	}
}
