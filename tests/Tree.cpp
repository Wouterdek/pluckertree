#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <iostream>
#include <Pluckertree.h>
#include <random>
#include "DataSetGenerator.h"

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

#include <chrono>

TEST(Tree, DISABLED_TestFindNeighbours_1_Random)
{
    for(int pass = 0; pass < 1; ++pass)
    {
        unsigned int line_count = 100000;
        unsigned int query_count = 100;

        std::random_device dev{};

        unsigned int line_seed = dev();
        std::cout << "Line generation seed: " << line_seed << std::endl;
        std::vector<LineWrapper> lines = GenerateRandomLines(line_seed, line_count, 100);

        auto tree = TreeBuilder<LineWrapper, &LineWrapper::l>::Build(lines.begin(), lines.end());

        unsigned int query_seed = dev();
        std::cout << "Query generation seed: " << query_seed << std::endl;
        auto query_points = GenerateRandomPoints(query_seed, query_count, 100);

        std::array<const LineWrapper *, 1> result{nullptr};

        int i = 0;
        for (const auto &query : query_points) {
            /*if (i < 8) {
                i++;
                continue;
            }*/
            //if(i % 10 == 0)
            {
                std::cout << i << std::endl;
            }
            //auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            //std::cout << "i: " << i << ", " << std::ctime(&t) << std::endl;
            float result_dist = 1E99;
            auto nbResultsFound = tree.FindNeighbours(query, result.begin(), result.end(), result_dist);
            std::cout << "visited nodes: " << Diag::visited << std::endl;
            EXPECT_EQ(nbResultsFound, 1);

            auto smallestLineIt = std::min_element(lines.begin(), lines.end(),
                                                   [query](const LineWrapper &l1, const LineWrapper &l2) {
                                                       auto l1Norm = (query.cross(l1.l.d) - l1.l.m).squaredNorm();
                                                       auto l2Norm = (query.cross(l2.l.d) - l2.l.m).squaredNorm();
                                                       return l1Norm < l2Norm;
                                                   });

            Line &SmallestLine = smallestLineIt->l;
            Vector3f m_spher = cart2spherical(SmallestLine.m);
            std::vector<int> idx;
            int sectI = 0;
            //std::cout << Diag::minimizations << std::endl;
            /*for (const auto &sector: tree.sectors) {
                if (Eigen::AlignedBox<float, 3>(sector.bounds.m_start, sector.bounds.m_end).contains(
                        cart2spherical(SmallestLine.m))
                    && sector.bounds.d_bound_1.dot(SmallestLine.d) >= 0 //greater or equal, or just greater?
                    && sector.bounds.d_bound_2.dot(SmallestLine.d) >= 0) {
                    idx.push_back(sectI);
                    const auto *node = sector.rootNode.get();
                    Bounds b1 = sector.bounds;
                    while (node != nullptr) {
                        //Bounds b2 = sector.bounds;
                        if(node->type == NodeType::moment)
                        {
                            b1.m_end[node->bound_component_idx] = node->m_component;
                            //b2.m_start[sector.rootNode->bound_component_idx] = sector.rootNode->m_component;
                        } else
                        {
                            b1.d_bound_1 = node->d_bound;
                            //b1.d_bound_2 = bounds.d_bound_2;
                        }


                        if (Eigen::AlignedBox<float, 3>(b1.m_start, b1.m_end).contains(cart2spherical(SmallestLine.m))
                            && b1.d_bound_1.dot(SmallestLine.d) >= 0 //greater or equal, or just greater?
                            && b1.d_bound_2.dot(SmallestLine.d) >= 0) {
                            idx.push_back(0);
                            node = node->children[0].get();
                        } else {
                            idx.push_back(1);
                            node = node->children[1].get();
                        }
                    }
                    break;
                }
                sectI++;
            }*/

            if (SmallestLine != result[0]->l) {
                std::cout << std::endl;
                std::cout << "sect: ";
                for (int curI : idx) {
                    std::cout << curI << " ";
                }
                std::cout << std::endl;
                std::cout << "Querypoint index: " << i << std::endl;
                std::cout << "Distance found: " << result_dist << std::endl;
                std::cout << "Actual smallest distance: " << (query.cross(SmallestLine.d) - SmallestLine.m).norm()
                          << std::endl;
                EXPECT_EQ(SmallestLine, result[0]->l);
            }
            i++;
        }
    }
}

TEST(Tree, DISABLED_ShowMeTheGrid)
{
    //dlb, dub, m_start, m_end

    Eigen::Vector3f q(1,0,0);
    Eigen::Vector3f dlb = Eigen::Vector3f(-1,0,0);
    Eigen::Vector3f dub = Eigen::Vector3f(0,-1,0);
    //top
    Eigen::Vector3f mlb(-M_PI, 0, 0);
    Eigen::Vector3f mub(M_PI, 0.785398185, 1);
    //left
    //Eigen::Vector3f mlb(M_PI/4, 0.785398185, 0);
    //Eigen::Vector3f mub(3*M_PI/4, 2.356194487, 1);
    //all
    //Eigen::Vector3f mlb(-M_PI, 0, 0);
    //Eigen::Vector3f mub(M_PI, M_PI, 1);
    //hemisphere
    //Eigen::Vector3f mlb(-M_PI, M_PI/2, 0);
    //Eigen::Vector3f mub(M_PI, M_PI, 1);
    std::string file = "/home/wouter/Desktop/pluckerdata/0";
    pluckertree::show_me_the_grid(file, dlb, dub, mlb, mub, q);
}

TEST(Tree, DISABLED_TestFindNearestHit_Random)
{
    for(int pass = 0; pass < 1; ++pass)
    {

    unsigned int line_count = 100;
    unsigned int query_count = 100;

    std::random_device dev{};

    unsigned int line_seed = dev();
    std::cout << "Line generation seed: " << line_seed << std::endl;
    std::vector<LineWrapper> lines = GenerateRandomLines(line_seed, line_count, 100);

    auto tree = TreeBuilder<LineWrapper, &LineWrapper::l>::Build(lines.begin(), lines.end());

    unsigned int query_seed = dev();
    std::cout << "Query generation seed: " << query_seed << std::endl;
    auto query_points = GenerateRandomPoints(query_seed, query_count, 100);
    auto query_point_normals = GenerateRandomNormals(query_seed, query_count);

    std::array<const LineWrapper*, 1> result { nullptr };

    int i = 0;
    for(unsigned int query_i = 0; query_i < query_points.size(); ++query_i)
    {
        const auto& query = query_points[query_i];
        const auto& query_normal = query_point_normals[query_i];

        //auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        //std::cout << "i: " << i << ", " << std::ctime(&t) << std::endl;
        float result_dist = 1E99;
        auto nbResultsFound = tree.FindNearestHits(query, query_normal, result.begin(), result.end(), result_dist);

        EXPECT_EQ(nbResultsFound, 1);

        auto smallestLineIt = std::min_element(lines.begin(), lines.end(), [query, query_normal](const LineWrapper& l1, const LineWrapper& l2){
            auto distF = [](const Eigen::Vector3f& p, const Eigen::Vector3f& n, const Line& l){
                Eigen::Vector3f l0 = (l.d.cross(l.m));
                Eigen::Vector3f intersection = l0 + (l.d * (p - l0).dot(n)/(l.d.dot(n)));
                Eigen::Vector3f vect = intersection - p;
                return vect;
            };

            auto l1Norm = distF(query, query_normal, l1.l).squaredNorm();
            auto l2Norm = distF(query, query_normal, l2.l).squaredNorm();
            return l1Norm < l2Norm;
        });

        Line& SmallestLine = smallestLineIt->l;
        Vector3f m_spher = cart2spherical(SmallestLine.m);
        std::vector<int> idx;
        int sectI = 0;
        /*for (const auto &sector: tree.sectors) {
            if (Eigen::AlignedBox<float, 3>(sector.bounds.m_start, sector.bounds.m_end).contains(
                    cart2spherical(SmallestLine.m))
                && sector.bounds.d_bound_1.dot(SmallestLine.d) >= 0 //greater or equal, or just greater?
                && sector.bounds.d_bound_2.dot(SmallestLine.d) >= 0) {
                idx.push_back(sectI);
                const auto *node = sector.rootNode.get();
                Bounds b1 = sector.bounds;
                while (node != nullptr) {
                    //Bounds b2 = sector.bounds;
                    if(node->type == NodeType::moment)
                    {
                        b1.m_end[node->bound_component_idx] = node->m_component;
                        //b2.m_start[sector.rootNode->bound_component_idx] = sector.rootNode->m_component;
                    } else
                    {
                        b1.d_bound_1 = node->d_bound;
                        //b1.d_bound_2 = bounds.d_bound_2;
                    }


                    if (Eigen::AlignedBox<float, 3>(b1.m_start, b1.m_end).contains(cart2spherical(SmallestLine.m))
                        && b1.d_bound_1.dot(SmallestLine.d) >= 0 //greater or equal, or just greater?
                        && b1.d_bound_2.dot(SmallestLine.d) >= 0) {
                        idx.push_back(0);
                        node = node->children[0].get();
                    } else {
                        idx.push_back(1);
                        node = node->children[1].get();
                    }
                }
                break;
            }
            sectI++;
        }*/

        if(SmallestLine != result[0]->l)
        {
            std::cout << "actual smallest: " ;
            float smallest = 1E99;
            /*for(float curVal : TreeNode::results)
            {
                smallest = std::min(smallest, curVal);
                //std::cout << curVal << " ";
            }*/
            std::cout << smallest << " ";
            //std::cout << "returned smallest: " << TreeNode::results.back() << std::endl;
            std::cout << std::endl;
            std::cout << "sect: " ;
            for(int curI : idx)
            {
                std::cout << curI << " ";
            }
            std::cout << std::endl;
            std::cout << "Querypoint index: " << i << std::endl;
            std::cout << "Distance found: " << result_dist << std::endl;
            std::cout << "Actual smallest distance: " << (query.cross(SmallestLine.d) - SmallestLine.m).norm() << std::endl;
            EXPECT_EQ(SmallestLine, result[0]->l);
        }
        i++;
    }

    }
}

unsigned int CountNodes(const std::unique_ptr<TreeNode<LineWrapper, &LineWrapper::l>>& n)
{
    if(n == nullptr)
    {
        return 0;
    }

    auto c1 = CountNodes(n->children[0]);
    auto c2 = CountNodes(n->children[1]);
    return 1 + c1 + c2;
};

TEST(Tree, DISABLED_TreeSize_Random)
{
    for(int pass = 0; pass < 100; ++pass)
    {
        unsigned int line_count = 100000;

        std::random_device dev{};

        unsigned int line_seed = dev();

        std::vector<LineWrapper> lines = GenerateRandomLines(line_seed, line_count, 100);

        auto tree = TreeBuilder<LineWrapper, &LineWrapper::l>::Build(lines.begin(), lines.end());

        unsigned int nodes = 0;
        for(const auto& sector : tree.sectors)
        {
            nodes += CountNodes(sector.rootNode);
        }

        if(tree.size() != line_count || tree.size() != nodes)
        {
            std::cout << "Line generation seed: " << line_seed << std::endl;
        }
        EXPECT_EQ(tree.size(), line_count);
        EXPECT_EQ(tree.size(), nodes);
    }
}
