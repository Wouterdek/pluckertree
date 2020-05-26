#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <Pluckertree.h>
#include <MathUtil.h>
#include <iostream>

using namespace testing;
using namespace Eigen;
using namespace pluckertree;

#define EXPECT_VEC3_NEAR(vect1, vect2, epsilon) \
    EXPECT_NEAR(vect1[0], vect2[0], epsilon); \
    EXPECT_NEAR(vect1[1], vect2[1], epsilon); \
    EXPECT_NEAR(vect1[2], vect2[2], epsilon)

TEST(RegionDistanceMinimizer, TestTopSector1)
{
    Eigen::Vector3f dlb = Eigen::Vector3f(0, 1, 0);
    Eigen::Vector3f dub = Eigen::Vector3f(0, 1, 0);
    Eigen::Vector3f mlb(-M_PI, 0, 1E-3);
    Eigen::Vector3f mub(M_PI, M_PI/4.0, 5);
    TreeNode node(dlb, dub, mlb, mub);

    Eigen::Vector3f q(1, 0, 0);

    Eigen::Vector3f min;
    auto minDist = node.CalcDistLowerBound(q, min);
    float epsilon = 1e-6;
    EXPECT_NEAR(minDist, 0, epsilon);
}

TEST(RegionDistanceMinimizer, TestTopSector2)
{
    Eigen::Vector3f dlb = Eigen::Vector3f(0, -1, 0);
    Eigen::Vector3f dub = Eigen::Vector3f(0, -1, 0);
    Eigen::Vector3f mlb(-M_PI, 0, 1);
    Eigen::Vector3f mub(M_PI, M_PI/4.0, 5);
    TreeNode node(dlb, dub, mlb, mub);

    Eigen::Vector3f q(1, 0, 0);

    Eigen::Vector3f min;
    auto minDist = node.CalcDistLowerBound(q, min);
    float epsilon = 1e-6;
    EXPECT_NEAR(minDist, 0.806143, epsilon);
}

TEST(RegionDistanceMinimizer, TestSideSector1)
{
    Eigen::Vector3f dlb = Eigen::Vector3f(0, 0, 1);
    Eigen::Vector3f dub = Eigen::Vector3f(0, 0, 1);
    Eigen::Vector3f mlb(-3*M_PI/4.0, 1*M_PI/4.0, 1);
    Eigen::Vector3f mub(-1*M_PI/4.0, 3*M_PI/4.0, 5);
    TreeNode node(dlb, dub, mlb, mub);

    Eigen::Vector3f q(1, 0, 0);

    Eigen::Vector3f min;
    auto minDist = node.CalcDistLowerBound(q, min);
    float epsilon = 1e-6;
    EXPECT_NEAR(minDist, 0, epsilon);
}

TEST(RegionDistanceMinimizer, TestSideSector2)
{
    Eigen::Vector3f dlb = Eigen::Vector3f(0, 0, 1);
    Eigen::Vector3f dub = Eigen::Vector3f(0, 0, 1);
    Eigen::Vector3f mlb(-3*M_PI/4.0, 1*M_PI/4.0, .1);
    Eigen::Vector3f mub(-1*M_PI/4.0, 3*M_PI/4.0, .5);
    TreeNode node(dlb, dub, mlb, mub);

    Eigen::Vector3f q(1, 0, 0);

    Eigen::Vector3f min;
    auto minDist = node.CalcDistLowerBound(q, min);
    float epsilon = 1e-6;
    EXPECT_NEAR(minDist, 0, epsilon);
}