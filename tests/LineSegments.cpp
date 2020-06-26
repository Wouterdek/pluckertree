#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <PluckertreeSegments.h>
#include <iostream>
#include <random>
#include "DataSetGenerator.h"

using namespace testing;
using namespace Eigen;

TEST(Tree, DISABLED_TestFindNeighbouringLineSegments_Random)
{
    for(int pass = 0; pass < 100; ++pass)
    {
        unsigned int line_count = 100;
        unsigned int query_count = 100;

        std::random_device dev{};

        unsigned int seed = dev();
        std::cout << "Line segment generation seed: " << seed << std::endl;
        std::vector<LineSegmentWrapper> lines = GenerateRandomLineSegments(dev, seed, line_count, 100, -100, 100);

        auto tree = pluckertree::segments::TreeBuilder<LineSegmentWrapper, &LineSegmentWrapper::l>::Build(lines.begin(), lines.end());

        std::vector<Vector3f> query_points{};
        {
            query_points.reserve(query_count);

            unsigned int seed = dev();
            std::cout << "Query generation seed: " << seed << std::endl;
            std::default_random_engine rng{seed};
            std::uniform_real_distribution<float> dist(0, 100);

            for (int i = 0; i < query_count; ++i) {
                query_points.emplace_back(dist(rng), dist(rng), dist(rng));
            }
        }

        std::array<const LineSegmentWrapper *, 1> result{nullptr};

        int i = 0;
        for (const auto &query : query_points) {
            if(i % 10 == 0)
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
                float t = p.dot(q);
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

TEST(Tree, TestFindNearestHitLineSegments_Random)
{
    for(int pass = 0; pass < 100; ++pass)
    {
        unsigned int line_count = 100;
        unsigned int query_count = 100;

        std::random_device dev{};

        unsigned int seed = dev();
        std::cout << "Line segment generation seed: " << seed << std::endl;
        std::vector<LineSegmentWrapper> lines = GenerateRandomLineSegments(dev, seed, line_count, 100, -100, 100);

        auto tree = pluckertree::segments::TreeBuilder<LineSegmentWrapper, &LineSegmentWrapper::l>::Build(lines.begin(), lines.end());

        std::vector<Vector3f> query_points {};
        std::vector<Vector3f> query_point_normals {};
        {
            query_points.reserve(query_count);

            unsigned int seed = dev();
            std::cout << "Query generation seed: " << seed << std::endl;
            std::default_random_engine rng{seed};
            std::uniform_real_distribution<float> dist(0, 100);

            for (int i = 0; i < query_count; ++i) {
                query_points.emplace_back(dist(rng), dist(rng), dist(rng));
                query_point_normals.emplace_back(dist(rng), dist(rng), dist(rng));
                query_point_normals.back().normalize();
            }
        }

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