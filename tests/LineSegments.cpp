#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <PluckertreeSegments.h>
#include <iostream>
#include <random>
#include "DataSetGenerator.h"

using namespace testing;
using namespace Eigen;

std::vector<int> FindSegment(const pluckertree::segments::TreeNode<LineSegmentWrapper, &LineSegmentWrapper::l>* node, const pluckertree::segments::Bounds& parentBounds, const LineSegment& smallestLine)
{
    if (node == nullptr) {
        return {};
    }else if(node->content.l == smallestLine)
    {
        return {0xFACE0FF};
    }

    std::vector<int> idx {};

    std::array<pluckertree::segments::Bounds, 2> childBounds {};

    for(int i = 0; i < 2; ++i) {
        childBounds[i] = parentBounds;

        if (node->type == pluckertree::segments::NodeType::moment) {
            if (i == 0) {
                childBounds[0].m_end[node->bound_component_idx] = node->m_component;
            } else {
                childBounds[1].m_start[node->bound_component_idx] = node->m_component;
            }
        } else if (node->type == pluckertree::segments::NodeType::direction) {
            if (i == 0) {
                childBounds[i].d_bound_1 = node->d_bound;
                //childBounds[i].d_bound_2 = curBounds.d_bound_2;
            } else {
                //childBounds[i].d_bound_1 = curBounds.d_bound_1;
                childBounds[i].d_bound_2 = -node->d_bound;
            }
        } else if (node->type == pluckertree::segments::NodeType::t) {
            if (i == 0) {
                childBounds[i].t1Min = node->c1t1;
                childBounds[i].t2Max = node->c1t2;
            } else {
                childBounds[i].t1Min = node->c2t1;
                childBounds[i].t2Max = node->c2t2;
            }
        }
    }

    std::vector<int> subIdx;
    if (node->type == pluckertree::segments::NodeType::t) {
        auto c1Idx = FindSegment(node->children[0].get(), childBounds[0], smallestLine);
        if(c1Idx.size() > 0 && c1Idx[c1Idx.size()-1] == 0xFACE0FF)
        {
            idx.push_back(0);
            subIdx = c1Idx;
        }else{
            auto c2Idx = FindSegment(node->children[1].get(), childBounds[1], smallestLine);
            if(c2Idx.size() > 0 && c2Idx[c2Idx.size()-1] == 0xFACE0FF)
            {
                idx.push_back(1);
                subIdx = c2Idx;
            }
        }
    } else {

        if (Eigen::AlignedBox<float, 3>(childBounds[0].m_start, childBounds[0].m_end).contains(
                cart2spherical(smallestLine.l.m))
            && childBounds[0].d_bound_1.dot(smallestLine.l.d) >= 0 //greater or equal, or just greater?
            && childBounds[0].d_bound_2.dot(smallestLine.l.d) >= 0) {
            idx.push_back(0);
            subIdx = FindSegment(node->children[0].get(), childBounds[0], smallestLine);
        } else {
            idx.push_back(1);
            subIdx = FindSegment(node->children[1].get(), childBounds[1], smallestLine);
        }
    }

    for (int curIdx : subIdx) {
        idx.push_back(curIdx);
    }

    return idx;
}

std::vector<int> FindSegment(const pluckertree::segments::Tree<LineSegmentWrapper, &LineSegmentWrapper::l>& tree, const LineSegment& smallestLine)
{
    std::vector<int> idx;
    int sectI = 0;
    for (const auto &sector: tree.sectors) {
        if (Eigen::AlignedBox<float, 3>(sector.bounds.m_start, sector.bounds.m_end).contains(
                cart2spherical(smallestLine.l.m))
            && sector.bounds.d_bound_1.dot(smallestLine.l.d) >= 0 //greater or equal, or just greater?
            && sector.bounds.d_bound_2.dot(smallestLine.l.d) >= 0) {
            idx.push_back(sectI);
            const auto *node = sector.rootNode.get();
            pluckertree::segments::Bounds curBounds = sector.bounds;

            auto result = FindSegment(node, curBounds, smallestLine);
            for(int curIdx : result)
            {
                idx.push_back(curIdx);
            }

            break;
        }
        sectI++;
    }
    return idx;
}

TEST(Tree, DISABLED_TestLineSegmentsInBounds)
{
    for(int pass = 0; pass < 100; ++pass)
    {
        unsigned int line_count = 100;

        std::random_device dev{};

        unsigned int line_seed = dev();
        std::cout << "Line segment generation seed: " << line_seed << std::endl;
        auto lines = GenerateRandomLineSegments(line_seed, line_count, 100, -100, 100);

        auto tree = pluckertree::segments::TreeBuilder<LineSegmentWrapper, &LineSegmentWrapper::l>::Build(lines.begin(), lines.end());

        std::array<const LineSegmentWrapper *, 1> result{nullptr};

        std::vector<std::tuple<const pluckertree::segments::TreeNode<LineSegmentWrapper, &LineSegmentWrapper::l>*, float, float>> todo;
        for(const auto& sector : tree.sectors)
        {
            if(sector.rootNode != nullptr)
            {
                todo.emplace_back(sector.rootNode.get(), sector.bounds.t1Min, sector.bounds.t2Max);
            }
        }
        while(!todo.empty())
        {
            const auto [curNode, t1, t2] = todo.back();
            todo.pop_back();

            EXPECT_GE(curNode->content.l.t1, t1);
            EXPECT_LE(curNode->content.l.t2, t2);

            bool isTSplitNode = curNode->type == pluckertree::segments::NodeType::t;

            if(curNode->children[0] != nullptr)
            {
                todo.emplace_back(curNode->children[0].get(), isTSplitNode ? curNode->c1t1 : t1, isTSplitNode ? curNode->c1t2 : t2);
            }
            if(curNode->children[1] != nullptr)
            {
                todo.emplace_back(curNode->children[1].get(), isTSplitNode ? curNode->c2t1 : t1, isTSplitNode ? curNode->c2t2 : t2);
            }
        }
    }
}

TEST(Tree, TestFindNeighbouringLineSegments_Random)
{
    for(int pass = 0; pass < 100; ++pass)
    {
        unsigned int line_count = 100;
        unsigned int query_count = 100;

        std::random_device dev{};

        unsigned int line_seed = dev();
        std::cout << "Line segment generation seed: " << line_seed << std::endl;
        auto lines = GenerateRandomLineSegments(line_seed, line_count, 100, -100, 100);

        auto tree = pluckertree::segments::TreeBuilder<LineSegmentWrapper, &LineSegmentWrapper::l>::Build(lines.begin(), lines.end());

        unsigned int query_seed = dev();
        std::cout << "Query generation seed: " << query_seed << std::endl;
        auto query_points = GenerateRandomPoints(query_seed, query_count, 100);

        std::array<const LineSegmentWrapper *, 1> result{nullptr};

        int i = 0;
        for (const auto &query : query_points) {
            if(i % 50 == 0)
            {
                std::cout << i << std::endl;
            }
            //auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            //std::cout << "i: " << i << ", " << std::ctime(&t) << std::endl;
            float result_dist = 1E99;
            auto nbResultsFound = tree.FindNeighbours(query, result.begin(), result.end(), result_dist);
            //std::cout << "visited nodes: " << TreeNode::visited << std::endl;
            EXPECT_EQ(nbResultsFound, 1);

            auto distF = [](const LineSegment& s, const Eigen::Vector3f& q)
            {
                Eigen::Vector3f p = s.l.d.cross(s.l.m);
                float t = s.l.d.dot(q);
                t = std::min(std::max(t, s.t1), s.t2);
                Eigen::Vector3f v = (p + t*s.l.d) - q;
                return v;
            };

            auto smallestLineIt = std::min_element(lines.begin(), lines.end(),
                                                   [query, distF](const LineSegmentWrapper &l1, const LineSegmentWrapper &l2) {
                                                       auto l1Norm = distF(l1.l, query).squaredNorm();
                                                       auto l2Norm = distF(l2.l, query).squaredNorm();
                                                       return l1Norm < l2Norm;
                                                   });

            LineSegment &SmallestLine = smallestLineIt->l;
            Vector3f m_spher = cart2spherical(SmallestLine.l.m);
            auto idx = FindSegment(tree, SmallestLine);

            if (SmallestLine != result[0]->l) {
                std::cout << std::endl;
                std::cout << "sect: ";
                for (int curI : idx) {
                    std::cout << curI << " ";
                }
                std::cout << std::endl;
                std::cout << "Querypoint index: " << i << std::endl;
                std::cout << "Distance found: " << result_dist << std::endl;
                std::cout << "Actual smallest distance: " << distF(SmallestLine, query).norm() << std::endl;
                EXPECT_EQ(SmallestLine, result[0]->l);
            }
            i++;
        }
    }
}

TEST(Tree, DISABLED_TestFindNearestHitLineSegments_Random)
{
    for(int pass = 0; pass < 100; ++pass)
    {
        unsigned int line_count = 100;
        unsigned int query_count = 100;

        std::random_device dev{};

        unsigned int line_seed = dev();
        std::cout << "Line segment generation seed: " << line_seed << std::endl;
        auto lines = GenerateRandomLineSegments(line_seed, line_count, 100, -100, 100);

        auto tree = pluckertree::segments::TreeBuilder<LineSegmentWrapper, &LineSegmentWrapper::l>::Build(lines.begin(), lines.end());

        unsigned int query_seed = dev();
        std::cout << "Query generation seed: " << query_seed << std::endl;
        auto query_points = GenerateRandomPoints(query_seed, query_count, 100);
        auto query_point_normals = GenerateRandomNormals(query_seed, query_count);

        std::array<const LineSegmentWrapper *, 1> result{nullptr};

        for(unsigned int query_i = 0; query_i < query_points.size(); ++query_i)
        {
            if(query_i % 10 == 0)
            {
                std::cout << query_i << std::endl;
            }

            const auto& query = query_points[query_i];
            const auto& query_normal = query_point_normals[query_i];

            //auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            //std::cout << "i: " << i << ", " << std::ctime(&t) << std::endl;
            float result_dist = 1E99;
            auto nbResultsFound = tree.FindNearestHits(query, query_normal, result.begin(), result.end(), result_dist);
            //std::cout << "visited nodes: " << TreeNode::visited << std::endl;

            auto distF = [&query_normal](const LineSegment& s, const Eigen::Vector3f& q)
            {
                Eigen::Vector3f l0 = (s.l.d.cross(s.l.m));
                auto t = (q - l0).dot(query_normal)/(s.l.d.dot(query_normal));
                if(t < s.t1 || t > s.t2)
                {
                    return 1E99f;
                }
                Eigen::Vector3f intersection = l0 + (s.l.d * t);
                Eigen::Vector3f vect = intersection - q;

                return vect.norm();
            };

            auto smallestLineIt = std::min_element(lines.begin(), lines.end(),
                                                   [query, distF](const LineSegmentWrapper &l1, const LineSegmentWrapper &l2) {
                                                       auto l1Norm = distF(l1.l, query);
                                                       auto l2Norm = distF(l2.l, query);
                                                       return l1Norm < l2Norm;
                                                   });

            LineSegment &SmallestLine = smallestLineIt->l;
            Vector3f m_spher = cart2spherical(SmallestLine.l.m);

            auto actual_smallest_dist = distF(SmallestLine, query);
            if(actual_smallest_dist == INFINITY)
            {
                EXPECT_EQ(nbResultsFound, 0);
            } else{
                EXPECT_EQ(nbResultsFound, 1);

                if (SmallestLine != result[0]->l && result_dist != actual_smallest_dist) {
                    std::cout << "Querypoint index: " << query_i << std::endl;
                    std::cout << "Distance found: " << result_dist << std::endl;
                    std::cout << "Actual smallest distance: " << actual_smallest_dist << std::endl;
                    EXPECT_EQ(SmallestLine, result[0]->l);
                }
            }
        }
    }
}