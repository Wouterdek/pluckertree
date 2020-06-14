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

struct LineWrapper
{
    Line l;
    explicit LineWrapper(Line line) : l(std::move(line)) {}
};

//100, 100, 3224486327, 1750131217, nlopt roundoff-limited
//100, 100, 4161057528, 2131736907, nlopt roundoff-limited
//100, 100, 729685792, 519425711
//100, 100, 3289432969, 3221991529

//100, 100, 240607058, 2271942277, nlopt roundoff-limited
//100, 100, 952045152, 903079720, nlopt roundoff-limited
//100, 100, 912149981, 2661143327, nlopt roundoff-limited
//100, 100, 3429696701, 2776366842, nlopt roundoff-limited
//100, 100, 702423470, 2130483690, nlopt roundoff-limited
//100, 100, 1276366762, 752999194, nlopt roundoff-limited
//100, 100, 3718611071, 759437136, nlopt roundoff-limited
//100, 100, 3769033040, 891816204

//100, 100, 738702163, 375110826
//100, 100, 2304416678, 672525704*
//100, 100, 1527464166, 910928879*
//100, 100, 1539785773, 2171327458
//100, 100, 1908813432, 3002281929
//100, 100, 3485206349, 3944654115
//100, 100, 2338550543, 1906307967 x2

//100, 100, 1562601595, 4055045319
//100, 100, 191214952, 933993246
//100, 100, 597655902, 93213975
//100, 100, 2491745788, 1514172478
//100, 100, 3496756572, 1131774831
//100, 100, 2127983384, 597997659
//100, 100, 1932884953, 3564452931 x2
#include <chrono>


TEST(Tree, DISABLED_TestFindNeighbours_1_Random)
{
    for(int pass = 0; pass < 100; ++pass)
    {
        unsigned int line_count = 100;
        unsigned int query_count = 100;

        std::random_device dev{};

        std::vector<LineWrapper> lines{};
        {
            lines.reserve(line_count);

            unsigned int seed = dev();
            std::cout << "Line generation seed: " << seed << std::endl;
            std::default_random_engine rng{seed};
            std::uniform_real_distribution<float> dist(0, 100);
            for (int i = 0; i < line_count; ++i) {
                Line l(Vector3f::Zero(), Vector3f::Zero());
                do {
                    l = Line::FromTwoPoints(
                            Vector3f(dist(rng), dist(rng), dist(rng)),
                            Vector3f(dist(rng), dist(rng), dist(rng))
                    );
                }while(l.m.norm() >= 150);
                lines.emplace_back(l);
            }
        }

        auto tree = TreeBuilder<LineWrapper, &LineWrapper::l>::Build(lines.begin(), lines.end());

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

        std::array<const LineWrapper *, 1> result{nullptr};

        int i = 0;
        for (const auto &query : query_points) {
            //if (i < 83) {
            //    i++;
            //    continue;
            //}
            if(i % 10 == 0)
            {
                //std::cout << i << std::endl;
            }
            //auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            //std::cout << "i: " << i << ", " << std::ctime(&t) << std::endl;
            float result_dist = 1E99;
            auto nbResultsFound = tree.FindNeighbours(query, result.begin(), result.end(), result_dist);
            //std::cout << "visited nodes: " << TreeNode::visited << std::endl;
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

    /*Eigen::Vector3f dlb = Eigen::Vector3f(0,0,-1);
    Eigen::Vector3f dub = Eigen::Vector3f(-std::sqrt(2)/2.0, std::sqrt(2)/2, 0);
    Eigen::Vector3f mlb(-M_PI, 0.785398185, 1);
    Eigen::Vector3f mub(-M_PI/2, 2.3561945, 80);
    Eigen::Vector3f q(39.110939, 52.7779579, 40.48032);*/
    Eigen::Vector3f dlb = Eigen::Vector3f(0.707106769, 0.707106769, 0);
    Eigen::Vector3f dub = Eigen::Vector3f(0,0,1);
    Eigen::Vector3f mlb(1.57079637, 0.785398185, 129.264511);
    Eigen::Vector3f mub(2.27183151, 1.74038839, 133.691605);
    Eigen::Vector3f q(28.2510929,16.3163109,34.3497543);
    std::string file = "/home/wouter/Desktop/pluckerdata/0";
    pluckertree::show_me_the_grid(file, dlb, dub, mlb, mub, q);
}

//3377894755, 4158056333
TEST(Tree, TestFindNearestHit_Random)
{
    for(int pass = 0; pass < 100; ++pass)
    {

    unsigned int line_count = 100;
    unsigned int query_count = 100;

    std::random_device dev {};

    std::vector<LineWrapper> lines {};
    {
        lines.reserve(line_count);

        unsigned int seed = dev();
        std::cout << "Line generation seed: " << seed << std::endl;
        std::default_random_engine rng {seed};
        std::uniform_real_distribution<float> dist(0, 100);
        for(int i = 0; i < line_count; ++i)
        {
            Line l(Vector3f::Zero(), Vector3f::Zero());
            do {
                l = Line::FromTwoPoints(
                        Vector3f(dist(rng), dist(rng), dist(rng)),
                        Vector3f(dist(rng), dist(rng), dist(rng))
                );
            }while(l.m.norm() >= 150);
            lines.emplace_back(l);
        }
    }

    auto tree = TreeBuilder<LineWrapper, &LineWrapper::l>::Build(lines.begin(), lines.end());

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
