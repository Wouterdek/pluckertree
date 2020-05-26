#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <Pluckertree.h>
#include <MathUtil.h>
#include <iostream>

using namespace testing;
using namespace Eigen;
using namespace pluckertree;

Vector3f AnyDirBound(const Vector3f& q, const Vector3f& moment)
{
    return moment.cross(q);
}

float FindMinDist(const Vector3f& q, const Vector3f& cartMoment)
{
    Vector3f b = AnyDirBound(q, cartMoment);
    return FindMinDist(q, b, b, cart2spherical(cartMoment));
}

TEST(RegionDistanceMinimizer, TestFixedMomentUnrestrictedDirQAtOneX)
{
    //for each k, calc dlb and dub by making sure dot product with both da and db is > 0.
    // (dlb = dub = (da + db)/2 ?)

    Vector3f q(1, 0, 0);
    float epsilon = 1e-6;

    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0,  1.0, 0.0)), 0, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0,  0.7, 0.0)), 0, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0,  1.5, 0.0)), 0.5, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 1.0,  0.0, 0.0)), std::sqrt(2), epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 2.0,  0.0, 0.0)), std::sqrt(5), epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0,  0.0, 1.0)), 0, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0,  0.0, 0.7)), 0, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0,  0.0, 1.5)), 0.5, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0,  0.5, 0.5)), 0, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0,  3.0, 4.0)), 4, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f(-1.0,  0.0, 0.0)), std::sqrt(2), epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f(-2.0,  0.0, 0.0)), std::sqrt(5), epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0, -1.0, 0.0)), 0, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0, -0.7, 0.0)), 0, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0, -1.5, 0.0)), 0.5, epsilon);
}

TEST(RegionDistanceMinimizer, TestFixedMomentUnrestrictedDirQAtTwoX)
{
    Vector3f q(2, 0, 0);
    float epsilon = 1e-6;

    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0, 1.0, 0.0)), 0, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0, 3.0, 0.0)), 1, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 1.0, 0.0, 0.0)), std::sqrt(5), epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 2.0, 0.0, 0.0)), std::sqrt(8), epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f(-1.0, 0.0, 0.0)), std::sqrt(5), epsilon);
}

TEST(RegionDistanceMinimizer, TestFixedMomentUnrestrictedDirQAtNegativeTwoX)
{
    Vector3f q(-2, 0, 0);
    float epsilon = 1e-6;

    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0,  1.0, 0.0)), 0, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0,  3.0, 0.0)), 1, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 1.0,  0.0, 0.0)), std::sqrt(5), epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 2.0,  0.0, 0.0)), std::sqrt(8), epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0,  0.0, 1.0)), 0, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0,  0.5, 0.5)), 0, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0,  3.0, 4.0)), 3, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f(-1.0,  0.0, 0.0)), std::sqrt(5), epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f(-2.0,  0.0, 0.0)), std::sqrt(8), epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0, -1.0, 0.0)), 0, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0, -2.0, 0.0)), 0, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0, -3.0, 0.0)), 1, epsilon);
}

TEST(RegionDistanceMinimizer, TestFixedMomentUnrestrictedDirQAtY)
{
    Vector3f q(0, 1, 0);
    float epsilon = 1e-6;

    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0,  1.0, 0.0)), std::sqrt(2), epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 1.0,  0.0, 0.0)), 0, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 2.0,  0.0, 0.0)), 1, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0,  0.0, 1.0)), 0, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0,  0.0, 0.7)), 0, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0,  0.0, 1.5)), 0.5, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0,  0.5, 0.5)), 1/std::sqrt(2), epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0,  3.0, 3.0)), std::sqrt(std::pow(1/std::sqrt(2), 2) + std::pow(std::sqrt(18)-(1/std::sqrt(2)), 2)), epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f(-1.0,  0.0, 0.0)), 0, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f(-2.0,  0.0, 0.0)), 1, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0, -1.0, 0.0)), std::sqrt(2), epsilon);
}

TEST(RegionDistanceMinimizer, TestFixedMomentUnrestrictedDirQMixed)
{
    Vector3f q(1, 1, 0);
    float epsilon = 1e-6;

    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0, 1.0, 0.0)), 1, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0, 0.7, 0.0)), 1, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0, 1.5, 0.0)), std::sqrt(0.5*0.5 + 1), epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 1.0, 0.0, 0.0)), 1, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 2.0, 0.0, 0.0)), std::sqrt(2), epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0, 0.0, 1.0)), 0, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0, 0.0, 0.7)), 0, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f( 0.0, 0.0, 1.5)), 1.5-std::sqrt(2), epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f(-1.0, 0.0, 0.0)), 1, epsilon);
}

TEST(RegionDistanceMinimizer, TestFixedMomentRestrictedDirQAtOneX)
{
    Vector3f q(1, 0, 0);
    float epsilon = 1e-6;

    EXPECT_NEAR(FindMinDist(q, Vector3f(0, 0, -1), Vector3f(0, 0, -1), cart2spherical(Vector3f(0.0, 1.0, 0.0))), 0, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f(0, 0, 1), Vector3f(0, 0, 1), cart2spherical(Vector3f(0.0, 1.0, 0.0))), 1, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f(1, 0, 1e-3).normalized(), Vector3f(-1, 0, 1e-3).normalized(), cart2spherical(Vector3f(0.0, 1.0, 0.0))), 2, epsilon); //Narrow bound here, but distance can technically be a bit lower than 2

    EXPECT_NEAR(FindMinDist(q, Vector3f(0.998455, 0, .055560).normalized(), Vector3f(-0.994729, 0, .102535).normalized(), cart2spherical(Vector3f(0.0, -1.0, 0.0))), 0, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f(0, 0, -1), Vector3f(0, 0, -1), cart2spherical(Vector3f(0.0, -1.0, 0.0))), 1, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f(1, 0, -1e-3).normalized(), Vector3f(-1, 0, -1e-3).normalized(), cart2spherical(Vector3f(0.0, -1.0, 0.0))), 2, epsilon); //Same here as above

    EXPECT_NEAR(FindMinDist(q, Vector3f(0, 1, 0), Vector3f(-1, 0, 0), cart2spherical(Vector3f(0.0, 0.0, 1.0))), 0, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f(0, -1, 0), Vector3f(0, -1, 0), cart2spherical(Vector3f(0.0, 0.0, 1.0))), 1, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f(0, -1, 0), Vector3f(-1, 0, 0), cart2spherical(Vector3f(0.0, 0.0, 1.0))), 1, epsilon);

    EXPECT_NEAR(FindMinDist(q, Vector3f(0, -1, 0), Vector3f(-1, 0, 0), cart2spherical(Vector3f(0.0, 0.0, -1.0))), 0, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f(0, 1, 0), Vector3f(0, 1, 0), cart2spherical(Vector3f(0.0, 0.0, -1.0))), 1, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f(0, -1, 0), Vector3f(0, -1, 0), cart2spherical(Vector3f(0.0, 0.0, -1.0))), 0, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f(1, .0001, 0).normalized(), Vector3f(-1, .0001, 0).normalized(), cart2spherical(Vector3f(0.0, 0.0, -1.0))), 2, epsilon);

    EXPECT_NEAR(FindMinDist(q, Vector3f(0, 0, 1), Vector3f(0, 1, 0), cart2spherical(Vector3f(1.0, 0.0, 0.0))), std::sqrt(2), epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f(0, 0, -1), Vector3f(0, 1, 0), cart2spherical(Vector3f(1.0, 0.0, 0.0))), std::sqrt(2), epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f(0, 0, 1), Vector3f(0, -1, 0), cart2spherical(Vector3f(1.0, 0.0, 0.0))), std::sqrt(2), epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f(0, 0, -1), Vector3f(0, -1, 0), cart2spherical(Vector3f(1.0, 0.0, 0.0))), std::sqrt(2), epsilon);
}

TEST(RegionDistanceMinimizer, TestFixedMomentRestrictedDirQAtTwoX)
{
    Vector3f q(2, 0, 0);
    float epsilon = 1e-6;

    EXPECT_NEAR(FindMinDist(q, Vector3f(1, 0, 0), Vector3f(0, 1, 0), cart2spherical(Vector3f(0.0, 0.0, 1.0))), 0, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f(-1, 0, 0), Vector3f(0, 1, 0), cart2spherical(Vector3f(0.0, 0.0, 1.0))), 0, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f(0, -1, 0), Vector3f(0, -1, 0), cart2spherical(Vector3f(0.0, 0.0, 1.0))), 1, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f(.5001, -.86603, 0).normalized(), Vector3f(-.5, .86603, 0).normalized(), cart2spherical(Vector3f(0.0, 0.0, 1.0))), 0, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f(1, 0, 0), Vector3f(0, -1, 0), cart2spherical(Vector3f(0.0, 0.0, 1.0))), 1, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f(-1, 0, 0), Vector3f(0, -1, 0), cart2spherical(Vector3f(0.0, 0.0, 1.0))), 1, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f(1, 0, 0), Vector3f(1, 0, 0), cart2spherical(Vector3f(0.0, 0.0, 1.0))), 0, epsilon);
    EXPECT_NEAR(FindMinDist(q, Vector3f(-1, 0, 0), Vector3f(-1, 0, 0), cart2spherical(Vector3f(0.0, 0.0, 1.0))), 0, epsilon);
}