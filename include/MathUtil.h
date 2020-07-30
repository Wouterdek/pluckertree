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
float calc_MAD(Iter begin, Iter end, Accessor f)
{
    std::sort(begin, end, [&f](const auto& v1, const auto& v2){
        return f(v1) < f(v2);
    });

    auto median = f(*(begin + ((end - begin)/2)));

    std::sort(begin, end, [&f, median](const auto& v1, const auto& v2){
        return std::abs(f(v1) - median) < std::abs(f(v2) - median);
    });

    auto mad = std::abs(f(*(begin + ((end - begin)/2))) - median);

    return mad;
}

template<typename Iter, typename Accessor>
float calc_pop_variance(Iter begin, Iter end, Accessor f)
{
    float total = 0;
    for(auto it = begin; it < end; ++it)
    {
        total += f(*it);
    }

    auto elemCount = std::distance(begin, end);
    float avg = total / elemCount;

    float variance = 0;
    for(auto it = begin; it < end; ++it)
    {
        float cur = f(*it) - avg;
        variance += cur * cur;
    }
    variance /= elemCount;

    return variance;
}

template<typename Iter, typename Accessor>
Eigen::Array3f calc_vec3_pop_variance(Iter begin, Iter end, Accessor f)
{
    Eigen::Array3f total(0, 0, 0);
    for(auto it = begin; it < end; ++it)
    {
        total += f(*it).array();
    }

    auto elemCount = std::distance(begin, end);
    Eigen::Array3f avg = total / elemCount;

    Eigen::Array3f variance(0, 0, 0);
    for(auto it = begin; it < end; ++it)
    {
        Eigen::Array3f cur = f(*it).array() - avg;
        variance += cur * cur;
    }
    variance /= elemCount;

    return variance;
}

