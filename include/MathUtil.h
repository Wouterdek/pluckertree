#pragma once

#include <Eigen/Dense>

template<typename number_t>
auto spherical2cart(number_t phi, number_t theta, number_t r)
{
    auto x = r * std::sin(theta) * std::cos(phi);
    auto y = r * std::sin(theta) * std::sin(phi);
    auto z = r * std::cos(theta);
    return Eigen::Matrix<number_t, 3, 1>(x, y, z);
}

template<typename Vector3_t>
auto cart2spherical(const Vector3_t& p)
{
    auto XsqPlusYsq = p[0]*p[0] + p[1]*p[1];
    auto r = p.norm();
    auto elev = std::atan2(std::sqrt(XsqPlusYsq), p[2]);
    auto az = std::atan2(p[1],p[0]);
    return Vector3_t(az, elev, r);
}

template<typename Iter, typename Accessor>
float calc_variance(Iter begin, Iter end, Accessor f)
{
    float total = 0;
    for(auto it = begin; it < end; ++it)
    {
        total += f(*it);
    }

    float avg = total / std::distance(begin, end);

    float variance = 0;
    for(auto it = begin; it < end; ++it)
    {
        float cur = f(*it) - avg;
        variance += cur * cur;
    }
    variance /= (std::distance(begin, end) - 1);

    return variance;
}

template<typename Iter, typename Accessor>
Eigen::Array3f calc_vec3_variance(Iter begin, Iter end, Accessor f)
{
    Eigen::Array3f total(0, 0, 0);
    for(auto it = begin; it < end; ++it)
    {
        total += f(*it);
    }

    Eigen::Array3f avg = total / std::distance(begin, end);

    Eigen::Array3f variance(0, 0, 0);
    for(auto it = begin; it < end; ++it)
    {
        Eigen::Array3f cur = f(*it) - avg;
        variance += cur * cur;
    }
    variance /= (std::distance(begin, end) - 1);

    return variance;
}