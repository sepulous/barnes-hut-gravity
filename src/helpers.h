#pragma once

#include <cmath>

struct Vec2
{
	double x;
	double y;

    double length() const
    {
        return std::sqrt(x*x + y*y);
    }

    double dot(const Vec2& other) const
    {
        return x * other.x + y * other.y;
    }

    Vec2& operator*=(double scalar)
    {
        x *= scalar;
        y *= scalar;
        return *this;
    }

    friend Vec2 operator*(double scalar, const Vec2& vec)
    {
        Vec2 new_vec = vec;
        new_vec.x *= scalar;
        new_vec.y *= scalar;
        return new_vec;
    }

    Vec2& operator+=(const Vec2& rhs)
    {
        x += rhs.x;
        y += rhs.y;
        return *this;
    }

    friend Vec2 operator+(Vec2 lhs, const Vec2& rhs)
    {
        lhs += rhs;
        return lhs;
    }

    Vec2& operator-=(const Vec2& rhs)
    {
        x -= rhs.x;
        y -= rhs.y;
        return *this;
    }

    friend Vec2 operator-(Vec2 lhs, const Vec2& rhs)
    {
        lhs -= rhs;
        return lhs;
    }
};

struct Point
{
    Vec2 position;
    Vec2 next_position;
    Vec2 velocity;
    Vec2 acceleration;
    double mass;
};
