#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <Pluckertree.h>
#include <MathUtil.h>
#include <iostream>

using namespace testing;
using namespace Eigen;
using namespace pluckertree;



TEST(RegionDistanceMinimizer, TestTopSector1)
{
    Eigen::Vector3f dlb = Eigen::Vector3f(0, 1, 0);
    Eigen::Vector3f dub = Eigen::Vector3f(0, 1, 0);
    Eigen::Vector3f mlb(-M_PI, 0, 1E-3);
    Eigen::Vector3f mub(M_PI, M_PI/4.0, 5);

    Eigen::Vector3f q(1, 0, 0);

    Eigen::Vector3f min;
    auto minDist = FindMinDist(q, dlb, dub, mlb, mub, min);
    float epsilon = 1e-6;
    EXPECT_NEAR(minDist, 0, epsilon);
}

TEST(RegionDistanceMinimizer, TestTopSector2)
{
    Eigen::Vector3f dlb = Eigen::Vector3f(0, -1, 0);
    Eigen::Vector3f dub = Eigen::Vector3f(0, -1, 0);
    Eigen::Vector3f mlb(-M_PI, 0, 1);
    Eigen::Vector3f mub(M_PI, M_PI/4.0, 5);

    Eigen::Vector3f q(1, 0, 0);

    Eigen::Vector3f min;
    auto minDist = FindMinDist(q, dlb, dub, mlb, mub, min);
    float epsilon = 1e-6;
    EXPECT_NEAR(minDist, 0.806143, epsilon);
}

TEST(RegionDistanceMinimizer, TestSideSector1)
{
    Eigen::Vector3f dlb = Eigen::Vector3f(0, 0, 1);
    Eigen::Vector3f dub = Eigen::Vector3f(0, 0, 1);
    Eigen::Vector3f mlb(-3*M_PI/4.0, 1*M_PI/4.0, 1);
    Eigen::Vector3f mub(-1*M_PI/4.0, 3*M_PI/4.0, 5);

    Eigen::Vector3f q(1, 0, 0);

    Eigen::Vector3f min;
    auto minDist = FindMinDist(q, dlb, dub, mlb, mub, min);
    float epsilon = 1e-6;
    EXPECT_NEAR(minDist, 0, epsilon);
}

TEST(RegionDistanceMinimizer, TestSideSector2)
{
    Eigen::Vector3f dlb = Eigen::Vector3f(0, 0, 1);
    Eigen::Vector3f dub = Eigen::Vector3f(0, 0, 1);
    Eigen::Vector3f mlb(-3*M_PI/4.0, 1*M_PI/4.0, .1);
    Eigen::Vector3f mub(-1*M_PI/4.0, 3*M_PI/4.0, .5);

    Eigen::Vector3f q(1, 0, 0);

    Eigen::Vector3f min;
    auto minDist = FindMinDist(q, dlb, dub, mlb, mub, min);
    float epsilon = 1e-6;
    EXPECT_NEAR(minDist, 0, epsilon);
}