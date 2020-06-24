#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <PluckertreeSegments.h>
#include <iostream>
#include <random>

using namespace testing;
using namespace Eigen;

using Line = pluckertree::Line;
using LineSegment = pluckertree::segments::LineSegment;

struct LineSegmentWrapper
{
    LineSegment l;
    explicit LineSegmentWrapper(LineSegment line) : l(std::move(line)) {}
};

TEST(Tree, TestFindNeighbouringLineSegments_Random)
{
    for(int pass = 0; pass < 100; ++pass)
    {
        unsigned int line_count = 100;
        unsigned int query_count = 100;

        std::random_device dev{};

        std::vector<LineSegmentWrapper> lines{};
        {
            lines.reserve(line_count);

            unsigned int seed = dev();
            std::cout << "Line segment generation seed: " << seed << std::endl;
            std::default_random_engine rng{seed};
            std::uniform_real_distribution<float> dist(0, 100);
            std::uniform_real_distribution<float> distT(-100, 100);
            for (int i = 0; i < line_count; ++i) {
                LineSegment l(Line(Vector3f::Zero(), Vector3f::Zero()), 0, 0);
                do {
                    float t1 = distT(rng);
                    float t2 = distT(rng);
                    if(t1 > t2)
                    {
                        std::swap(t1, t2);
                    }

                    l = LineSegment(Line::FromTwoPoints(
                            Vector3f(dist(rng), dist(rng), dist(rng)),
                            Vector3f(dist(rng), dist(rng), dist(rng))
                    ), t1, t2);
                }while(l.l.m.norm() >= 150);
                lines.emplace_back(l);
            }
        }

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
            //if (i < 83) {
            //    i++;
            //    continue;
            //}
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