#pragma once

#define EXPECT_VEC3_NEAR(vect1, vect2, epsilon) \
    EXPECT_NEAR(vect1[0], vect2[0], epsilon); \
    EXPECT_NEAR(vect1[1], vect2[1], epsilon); \
    EXPECT_NEAR(vect1[2], vect2[2], epsilon)
