#pragma once

#include <vector>
#include <memory>
#include <array>

#include "helpers.h"

class QuadTree
{
public:
	QuadTree();
	QuadTree(Vec2, Vec2);
	~QuadTree() = default;

	QuadTree(const QuadTree&) = delete;
	QuadTree& operator=(const QuadTree&) = delete;

	QuadTree(QuadTree&&) = default;
	QuadTree& operator=(QuadTree&&) = default;

	void Add(Point*);
	void CalculateMass();
	Vec2 GetCenter() const;
	Vec2 GetExtents() const;
	Vec2 GetCenterOfMass() const;
	double GetTotalMass() const;
	std::vector<QuadTree>& GetChildren();
	bool HasChildren() const;
	Point* GetPoint() const;
	void Print(int level = 0);

private:
	void AddToChild(Point*);
	void CreateChildren();

private:
	Point* point_ = nullptr;
	std::vector<QuadTree> children_;
	Vec2 center_;
	Vec2 extents_;
	Vec2 center_of_mass_;
	double total_mass_;
};
