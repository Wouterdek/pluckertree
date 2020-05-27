#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <Pluckertree.h>
#include "TestUtilities.h"

using namespace testing;
using namespace pluckertree;
using namespace Eigen;

TEST(Line, TestFromPointAndDirection1)
{
    auto epsilon = 1e-6;
    auto l = Line::FromPointAndDirection(Vector3f(0, 0, 1), Vector3f(1, 0, 0));
    EXPECT_VEC3_NEAR(l.d, Vector3f(1, 0, 0), epsilon);
    EXPECT_VEC3_NEAR(l.m, Vector3f(M_PI/2, M_PI/2, 1), epsilon);
}

TEST(Line, TestFromPointAndDirection2)
{
    auto epsilon = 1e-6;
    auto l = Line::FromPointAndDirection(Vector3f(2, 0, 0), Vector3f(0, 1, 0));
    EXPECT_VEC3_NEAR(l.d, Vector3f(0, 1, 0), epsilon);
    EXPECT_NEAR(l.m[1], 0, epsilon);
    EXPECT_NEAR(l.m[2], 2, epsilon);
}

TEST(Line, TestFromPointAndDirection3)
{
    auto epsilon = 1e-6;
    auto l = Line::FromPointAndDirection(Vector3f(1, 0, 1), Vector3f(1, 0, 0));
    EXPECT_VEC3_NEAR(l.d, Vector3f(1, 0, 0), epsilon);
    EXPECT_VEC3_NEAR(l.m, Vector3f(M_PI/2, M_PI/2, 1), epsilon);
}

TEST(Line, TestFromPointAndDirection4)
{
    auto epsilon = 1e-6;
    auto l = Line::FromPointAndDirection(Vector3f(-1, 0, 0), Vector3f(0, 0, 1));
    EXPECT_VEC3_NEAR(l.d, Vector3f(0, 0, 1), epsilon);
    EXPECT_VEC3_NEAR(l.m, Vector3f(M_PI/2, M_PI/2, 1), epsilon);
}

TEST(Line, TestFromPointAndDirection5)
{
    auto epsilon = 1e-6;
    Vector3f d = Vector3f(-1/std::sqrt(2.0f), -1/std::sqrt(2.0f), 1).normalized();
    auto l = Line::FromPointAndDirection(Vector3f(1, -1, 0).normalized(), d);
    EXPECT_VEC3_NEAR(l.d, d, epsilon);
    EXPECT_VEC3_NEAR(l.m, Vector3f(-3*M_PI/4, 3*M_PI/4, 1), epsilon);
}

TEST(Line, TestFromTwoPoints1)
{
    auto epsilon = 1e-6;
    auto l = Line::FromTwoPoints(Vector3f(0, 0, 1), Vector3f(1, 0, 1));
    EXPECT_VEC3_NEAR(l.d, Vector3f(1, 0, 0), epsilon);
    EXPECT_VEC3_NEAR(l.m, Vector3f(M_PI/2, M_PI/2, 1), epsilon);
}

TEST(Line, TestFromTwoPoints2)
{
    auto epsilon = 1e-6;
    auto l = Line::FromTwoPoints(Vector3f(2, 0, 0), Vector3f(2, 2, 0));
    EXPECT_VEC3_NEAR(l.d, Vector3f(0, 1, 0), epsilon);
    EXPECT_NEAR(l.m[1], 0, epsilon);
    EXPECT_NEAR(l.m[2], 2, epsilon);
}

TEST(Line, TestFromTwoPoints3)
{
    auto epsilon = 1e-6;
    auto l = Line::FromTwoPoints(Vector3f(1, 0, 1), Vector3f(2, 0, 1));
    EXPECT_VEC3_NEAR(l.d, Vector3f(1, 0, 0), epsilon);
    EXPECT_VEC3_NEAR(l.m, Vector3f(M_PI/2, M_PI/2, 1), epsilon);
}