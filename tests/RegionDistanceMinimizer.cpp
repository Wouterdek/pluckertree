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

    Eigen::Vector3f min = mlb + (mub - mlb)/2;
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

    Eigen::Vector3f min = mlb + (mub - mlb)/2;
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

    Eigen::Vector3f min = mlb + (mub - mlb)/2;
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

    Eigen::Vector3f min = mlb + (mub - mlb)/2;
    auto minDist = FindMinDist(q, dlb, dub, mlb, mub, min);
    float epsilon = 1e-6;
    EXPECT_NEAR(minDist, 0, epsilon);
}

TEST(RegionDistanceMinimizer, Test3)
{
    Eigen::Vector3f dlb = Eigen::Vector3f(0.707106769, 0.707106769, 0);
    Eigen::Vector3f dub = Eigen::Vector3f(0, 0, 1);
    Eigen::Vector3f mlb(1.57079637, 0.785398185, 0);
    Eigen::Vector3f mub(3.14159274, 2.3561945, 150);

    Eigen::Vector3f q(46.7477722, 45.1327858, 26.5332966);

    Eigen::Vector3f min = mlb + (mub - mlb)/2;
    auto minDist = FindMinDist(q, dlb, dub, mlb, mub, min);
    std::cout << "min: " << min.transpose() << std::endl;
    EXPECT_LT(minDist, 10);
}

TEST(RegionDistanceMinimizer, Test5)
{
    Eigen::Vector3f dlb = Eigen::Vector3f(0,0,-1);
    Eigen::Vector3f dub = Eigen::Vector3f(-std::sqrt(2)/2.0, std::sqrt(2)/2, 0);
    Eigen::Vector3f mlb(-1.670233346304141, 1.3752921333858823, 32.70795644289806);
    Eigen::Vector3f mub(-1.670233346304141, 1.3752921333858823, 32.70795644289806);

    Eigen::Vector3f q(20.1563377, 1.75047028, 48.4263535);
    Eigen::Vector3f q_normal(0.403123677, 0.569172323, 0.716612995);

    Eigen::Vector3f min = mlb + (mub - mlb)/2;
    auto minDist = FindMinHitDist(q, q_normal, dlb, dub, mlb, mub, min);
    std::cout << "min: " << min.transpose() << std::endl;
    EXPECT_LT(minDist, 6.45);
}

TEST(RegionDistanceMinimizer, Test6)
{
    Eigen::Vector3f dlb = Eigen::Vector3f(0,0,-1);
    Eigen::Vector3f dub = Eigen::Vector3f(-std::sqrt(2)/2.0, std::sqrt(2)/2, 0);
    Eigen::Vector3f mlb(-2.9656632640907143, 0.9729883518356381, 60.05482929947763);
    Eigen::Vector3f mub(-2.9656632640907143, 0.9729883518356381, 60.05482929947763);

    Eigen::Vector3f q(25.216011, 86.2393799, 64.2581253);
    Eigen::Vector3f q_normal(0.742901862, 0.662636876, 0.0949169919);

    Eigen::Vector3f min = mlb + (mub - mlb)/2;
    auto minDist = FindMinHitDist(q, q_normal, dlb, dub, mlb, mub, min);
    std::cout << "min: " << min.transpose() << std::endl;
    EXPECT_LT(minDist, 4.5);
}