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

//100, 100, 3224486327, 1750131217, nlopt roundoff-limited
//100, 100, 729685792, 519425711
//100, 100, 3289432969, 3221991529
//100, 100, 1534475676, 2371443842
#include <chrono>

TEST(Tree, DISABLED_TestFindNeighbours_1_Random)
{
    unsigned int line_count = 100;
    unsigned int query_count = 100;

    std::random_device dev {};

    std::vector<Line> lines {};
    {
        lines.reserve(line_count);

        unsigned int seed = dev();
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

        unsigned int seed = dev();
        std::cout << "Query generation seed: " << seed << std::endl;
        std::default_random_engine rng {seed};
        std::uniform_real_distribution<float> dist(0, 100);

        for(int i = 0; i < query_count; ++i)
        {
            query_points.emplace_back(dist(rng), dist(rng), dist(rng));
        }
    }

    std::array<const Line*, 1> result { nullptr };

    int i = 0;
    for(const auto& query : query_points)
    {
        //auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        //std::cout << "i: " << i << ", " << std::ctime(&t) << std::endl;
        float result_dist = 1E99;
        TreeNode::results.clear();
        auto nbResultsFound = tree.FindNeighbours(query, result.begin(), result.end(), result_dist);
        std::cout << "visited nodes: " << TreeNode::visited << std::endl;
        EXPECT_EQ(nbResultsFound, 1);

        auto smallestLineIt = std::min_element(lines.begin(), lines.end(), [query](const Line& l1, const Line& l2){
            auto l1Norm = (query.cross(l1.d) - l1.m).squaredNorm();
            auto l2Norm = (query.cross(l2.d) - l2.m).squaredNorm();
            return l1Norm < l2Norm;
        });

        Line SmallestLine = *smallestLineIt;
        Vector3f m_spher = cart2spherical(SmallestLine.m);
        std::vector<int> idx;
        int sectI = 0;
        for(const auto& sector: tree.sectors)
        {
            if(Eigen::AlignedBox<float, 3>(sector.bounds.m_start, sector.bounds.m_end).contains(cart2spherical(SmallestLine.m))
               && sector.bounds.d_bound_1.dot(SmallestLine.d) >= 0 //greater or equal, or just greater?
               && sector.bounds.d_bound_2.dot(SmallestLine.d) >= 0)
            {
                idx.push_back(sectI);
                const TreeNode* node = sector.rootNode.get();
                Bounds b1 = sector.bounds;
                while(node != nullptr)
                {
                    //Bounds b2 = sector.bounds;
                    b1.m_end[node->bound_component_idx] = node->m_component;
                    //b2.m_start[sector.rootNode->bound_component_idx] = sector.rootNode->m_component;
                    if(Eigen::AlignedBox<float, 3>(b1.m_start, b1.m_end).contains(cart2spherical(SmallestLine.m))
                       && b1.d_bound_1.dot(SmallestLine.d) >= 0 //greater or equal, or just greater?
                       && b1.d_bound_2.dot(SmallestLine.d) >= 0)
                    {
                        idx.push_back(0);
                        node = node->children[0].get();
                    }else{
                        idx.push_back(1);
                        node = node->children[1].get();
                    }
                }
                break;
            }
            sectI++;
        }

        if(*smallestLineIt != *result[0])
        {
            std::cout << "progress: " ;
            for(float curVal : TreeNode::results)
            {
                std::cout << curVal << " ";
            }
            std::cout << std::endl;
            std::cout << "sect: " ;
            for(int curI : idx)
            {
                std::cout << curI << " ";
            }
            std::cout << std::endl;
            std::cout << "Querypoint index: " << i << std::endl;
            std::cout << "Distance found: " << result_dist << std::endl;
            std::cout << "Actual smallest distance: " << (query.cross(smallestLineIt->d) - smallestLineIt->m).norm() << std::endl;
            ASSERT_EQ(*smallestLineIt, *result[0]);
        }
        i++;
    }
}

TEST(Tree, TestFindNearestHit_Random)
{
    unsigned int line_count = 100;
    unsigned int query_count = 100;

    std::random_device dev {};

    std::vector<Line> lines {};
    {
        lines.reserve(line_count);

        unsigned int seed = dev();
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
    std::vector<Vector3f> query_point_normals {};
    {
        query_points.reserve(query_count);
        query_point_normals.reserve(query_count);

        unsigned int seed = dev();
        std::cout << "Query generation seed: " << seed << std::endl;
        std::default_random_engine rng {seed};
        std::uniform_real_distribution<float> dist(0, 100);

        for(int i = 0; i < query_count; ++i)
        {
            query_points.emplace_back(dist(rng), dist(rng), dist(rng));
            query_point_normals.emplace_back(dist(rng), dist(rng), dist(rng));
            query_point_normals.back().normalize();
        }
    }

    std::array<const Line*, 1> result { nullptr };

    int i = 0;
    for(unsigned int query_i = 0; query_i < query_points.size(); ++query_i)
    {
        const auto& query = query_points[query_i];
        const auto& query_normal = query_point_normals[query_i];

        //auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        //std::cout << "i: " << i << ", " << std::ctime(&t) << std::endl;
        float result_dist = 1E99;
        TreeNode::results.clear();
        auto nbResultsFound = tree.FindNearestHits(query, query_normal, result.begin(), result.end(), result_dist);
        std::cout << "visited nodes: " << TreeNode::visited << std::endl;
        EXPECT_EQ(nbResultsFound, 1);

        auto smallestLineIt = std::min_element(lines.begin(), lines.end(), [query, query_normal](const Line& l1, const Line& l2){
            auto distF = [](const Eigen::Vector3f& p, const Eigen::Vector3f& n, const Line& l){
                Eigen::Vector3f l0 = (l.d.cross(l.m));
                Eigen::Vector3f intersection = l0 + (l.d * (p - l0).dot(n)/(l.d.dot(n)));
                Eigen::Vector3f vect = intersection - p;
                return vect;
            };

            auto l1Norm = distF(query, query_normal, l1).squaredNorm();
            auto l2Norm = distF(query, query_normal, l2).squaredNorm();
            return l1Norm < l2Norm;
        });

        Line SmallestLine = *smallestLineIt;
        Vector3f m_spher = cart2spherical(SmallestLine.m);
        std::vector<int> idx;
        int sectI = 0;
        for(const auto& sector: tree.sectors)
        {
            if(Eigen::AlignedBox<float, 3>(sector.bounds.m_start, sector.bounds.m_end).contains(cart2spherical(SmallestLine.m))
               && sector.bounds.d_bound_1.dot(SmallestLine.d) >= 0 //greater or equal, or just greater?
               && sector.bounds.d_bound_2.dot(SmallestLine.d) >= 0)
            {
                idx.push_back(sectI);
                const TreeNode* node = sector.rootNode.get();
                Bounds b1 = sector.bounds;
                while(node != nullptr)
                {
                    //Bounds b2 = sector.bounds;
                    b1.m_end[node->bound_component_idx] = node->m_component;
                    //b2.m_start[sector.rootNode->bound_component_idx] = sector.rootNode->m_component;
                    if(Eigen::AlignedBox<float, 3>(b1.m_start, b1.m_end).contains(cart2spherical(SmallestLine.m))
                       && b1.d_bound_1.dot(SmallestLine.d) >= 0 //greater or equal, or just greater?
                       && b1.d_bound_2.dot(SmallestLine.d) >= 0)
                    {
                        idx.push_back(0);
                        node = node->children[0].get();
                    }else{
                        idx.push_back(1);
                        node = node->children[1].get();
                    }
                }
                break;
            }
            sectI++;
        }

        if(*smallestLineIt != *result[0])
        {
            std::cout << "progress: " ;
            for(float curVal : TreeNode::results)
            {
                std::cout << curVal << " ";
            }
            std::cout << std::endl;
            std::cout << "sect: " ;
            for(int curI : idx)
            {
                std::cout << curI << " ";
            }
            std::cout << std::endl;
            std::cout << "Querypoint index: " << i << std::endl;
            std::cout << "Distance found: " << result_dist << std::endl;
            std::cout << "Actual smallest distance: " << (query.cross(smallestLineIt->d) - smallestLineIt->m).norm() << std::endl;
            ASSERT_EQ(*smallestLineIt, *result[0]);
        }
        i++;
    }
}
