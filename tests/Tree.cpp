#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <Pluckertree.h>
#include <iostream>
#include <random>

using namespace testing;
using namespace pluckertree;
using namespace Eigen;

/*
TEST(Tree, TestBuildTree_OneInEachSector)
{
    std::vector<Line> lines {
        Line::FromPointAndDirection(Vector3f(), Vector3f()),
        Line::FromPointAndDirection(Vector3f(), Vector3f()),
    };

    auto tree = pluckertree::TreeBuilder::Build(lines.begin(), lines.end());
}

TEST(Tree, TestBuildTree_TopSector)
{
    std::vector<Line> lines {
            Line::FromPointAndDirection(Vector3f(), Vector3f()),
            Line::FromPointAndDirection(Vector3f(), Vector3f()),
    };

    auto tree = pluckertree::TreeBuilder::Build(lines.begin(), lines.end());
}

TEST(Tree, TestBuildTree_XYSector)
{
    std::vector<Line> lines {
            Line::FromPointAndDirection(Vector3f(), Vector3f()),
            Line::FromPointAndDirection(Vector3f(), Vector3f()),
    };

    auto tree = pluckertree::TreeBuilder::Build(lines.begin(), lines.end());
}

TEST(Tree, TestBuildTree_SubNodesInParent)
{
    std::vector<Line> lines {
            Line::FromPointAndDirection(Vector3f(), Vector3f()),
            Line::FromPointAndDirection(Vector3f(), Vector3f()),
    };

    auto tree = pluckertree::TreeBuilder::Build(lines.begin(), lines.end());
}*/


TEST(Tree, TestFindNeighbours)
{
}

//10, 10, 350495777, 511987587
//100, 100, 2409447823, 3164519302
//100, 100, 2884071531, 4095387696
//100, 100, 34573069, 2478727349
//25, 25, 2061222708, 1977287255

TEST(Tree, TestFindNeighbours_1_Random)
{
    unsigned int line_count = 100;
    unsigned int query_count = 100;

    std::random_device dev {};

    std::vector<Line> lines {};
    {
        lines.reserve(line_count);

        unsigned int seed = 34573069;//dev();
        std::cout << "Line generation seed: " << seed << std::endl;
        std::default_random_engine rng {seed};
        std::uniform_real_distribution<float> dist(0, 100);
        for(int i = 0; i < line_count; ++i)
        {
            lines.push_back(Line::FromTwoPoints(
                    Vector3f(dist(rng), dist(rng), dist(rng)),
                    Vector3f(dist(rng), dist(rng), dist(rng))
            ));
        }
    }

    auto tree = TreeBuilder::Build(lines.begin(), lines.end());

    std::vector<Vector3f> query_points {};
    {
        query_points.reserve(query_count);

        unsigned int seed = 2478727349;//dev();
        std::cout << "Query generation seed: " << seed << std::endl;
        std::default_random_engine rng {seed};
        std::uniform_real_distribution<float> dist(0, 100);

        for(int i = 0; i < query_count; ++i)
        {
            query_points.emplace_back(dist(rng), dist(rng), dist(rng));
        }
    }

    std::array<const Line*, 1> result { nullptr };

    float result_dist = 1E99;
    for(const auto& query : query_points)
    {
        auto nbResultsFound = tree.FindNeighbours(query, result.begin(), result.end(), result_dist);
        EXPECT_EQ(nbResultsFound, 1);

        auto smallestLineIt = std::min_element(lines.begin(), lines.end(), [query](const Line& l1, const Line& l2){
            auto l1Norm = (query.cross(l1.d) - l1.m).squaredNorm();
            auto l2Norm = (query.cross(l2.d) - l2.m).squaredNorm();
            return l1Norm < l2Norm;
        });

        if(*smallestLineIt != *result[0])
        {
            std::cout << "Distance found: " << result_dist << std::endl;
            std::cout << "Actual smallest distance: " << (query.cross(smallestLineIt->d) - smallestLineIt->m).norm() << std::endl;
            ASSERT_EQ(&*smallestLineIt, result[0]);
        }
    }
}
