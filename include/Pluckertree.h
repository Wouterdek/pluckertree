#pragma once

#include <limits>
#include <cstdint>
#include <numeric>
#include <Eigen/Dense>
#include "MathUtil.h"

namespace pluckertree
{

/**
 * Find the moment vector that produces the line with the smallest distance to the querypoint.
 * phi = polar angle, theta = azimuth, r = radius.
 * @param point the querypoint
 * @param dirLowerBound lower bound of the directional vector, in spherical coordinates [phi; theta]
 * @param dirUpperBound upper bound of the directional vector, in spherical coordinates [phi; theta]
 * @param momentLowerBound lower bound of the moment vector, in spherical coordinates [phi; theta; r]
 * @param momentUpperBound upper bound of the moment vector, in spherical coordinates [phi; theta; r]
 * @param min output parameter, will receive the moment vector
 * @return the distance from the line to the querypoint
 */
double FindMinDist(
        const Eigen::Vector3f& point,
        const Eigen::Vector3f& dirLowerBound,
        const Eigen::Vector3f& dirUpperBound,
        const Eigen::Vector3f& momentLowerBound,
        const Eigen::Vector3f& momentUpperBound,
        Eigen::Vector3f& min
);

double FindMinDist(
        const Eigen::Vector3f& point,
        const Eigen::Vector3f& dirLowerBound,
        const Eigen::Vector3f& dirUpperBound,
        const Eigen::Vector3f& moment
);

void show_me_the_grid();

class Line
{
public:
	Line(Eigen::Vector3f d, Eigen::Vector3f m) : d(std::move(d)), m(std::move(m)) {};

	static Line FromPointAndDirection(const Eigen::Vector3f& point, const Eigen::Vector3f& direction)
	{
	    auto p = point - point.dot(direction)*direction;
	    auto m = p.cross(direction);
	    return Line(direction, cart2spherical(m));

	    //Alternatively:
	    auto dot = point.dot(direction);
	    auto pointSqrNorm = point.squaredNorm();
	    auto scale = std::sqrt(pointSqrNorm) / std::sqrt(dot*dot + pointSqrNorm);
        return Line(direction, point.cross(direction) * scale);
	}

	static Line FromTwoPoints(const Eigen::Vector3f& point_a, const Eigen::Vector3f& point_b)
	{
        return FromPointAndDirection(point_a, (point_b-point_a).normalized());
	}

    Eigen::Vector3f d;
	Eigen::Vector3f m; // In spherical coordinates (azimuth, elevation with 0=Z+, radius)
};

class TreeNode;

struct Bounds
{
    Eigen::Vector3f d_bound_1;
    Eigen::Vector3f d_bound_2;
    Eigen::Vector3f m_start;
    Eigen::Vector3f m_end;

    Bounds(Eigen::Vector3f d_bound_1, Eigen::Vector3f d_bound_2, Eigen::Vector3f m_start, Eigen::Vector3f m_end)
    : d_bound_1(std::move(d_bound_1)), d_bound_2(std::move(d_bound_2)), m_start(std::move(m_start)), m_end(std::move(m_end)) {}

    Bounds() = default;
};

class TreeSector
{
public:
    Bounds bounds;
    std::unique_ptr<TreeNode> rootNode;

    TreeSector(Eigen::Vector3f d_bound_1, Eigen::Vector3f d_bound_2, Eigen::Vector3f m_start, Eigen::Vector3f m_end)
        : bounds(std::move(d_bound_1), std::move(d_bound_2), std::move(m_start), std::move(m_end)) {}
};

enum class NodeType : uint8_t
{
    moment, direction
};

/**
 *  @brief  Inserts given value into the iterator range before specified position.
 *  @param  __position  The position.
 *  @param  __x  Data to be inserted.
 *  @return  An iterator that points to the inserted data.
 *
 *  This function will insert a copy of the given value before the specified location.
 *  All values will shift towards the end, with the last element being discarded.
 *  position must be smaller than end.
 */
template<typename Iterator, typename Value, typename = typename std::enable_if<std::is_same<Value, typename Iterator::value_type>::value>::type>
Iterator iter_insert(Iterator position, Iterator end, const Value& val)
{
    assert(position < end);

    for(Iterator it = end-2; it >= position; --it)
    {
        *(it+1) = *it;
    }
    *position = val;
}

class TreeNode
{
public:
    std::array<std::unique_ptr<TreeNode>, 2> children;
    Line line;
    Eigen::Vector3f d_bound;
    NodeType type : 1;
    //uint8_t bound_idx : 1;
    uint8_t bound_component_idx : 2;
    //uint8_t pad1 : 6;
    //uint8_t pad2[7];

    TreeNode(uint8_t bound_component_idx, Line line)
        : type(NodeType::moment), bound_component_idx(bound_component_idx), line(std::move(line)), children() {}

    TreeNode(Eigen::Vector3f d_bound, Line line)
            : type(NodeType::direction), d_bound(std::move(d_bound)), line(std::move(line)), children() {}

    /*double CalcDistLowerBound(const Eigen::Vector3f& q, Eigen::Vector3f& optimal_moment)
    {
        if(type == NodeType::moment)
        {
            return FindMinDist(q, this->bound_1, this->bound_2, this->m_start, this->m_end, optimal_moment);
        }
        else if(type == NodeType::direction)
        {

        }
    }*/

    template<class OutputIt, typename = typename std::enable_if<std::is_same<Line, typename OutputIt::value_type>::value>::type>
    typename std::iterator_traits<OutputIt>::difference_type FindNeighbours(
            const Eigen::Vector3f& query_point,
            OutputIt out_first, OutputIt out_last,
            float& max_dist,
            const Bounds& bounds
    ) const
    {
        std::array<float, 2> minimumDistances {};
        std::array<Bounds, 2> childBounds {};

        for(int i = 0; i < 2; ++i)
        {
            const auto& c = children[i];
            if(c == nullptr)
            {
                minimumDistances[i] = std::numeric_limits<float>::infinity();
            } else
            {
                childBounds[i] = bounds;

                if(this->type == NodeType::moment)
                {
                    if(i == 0)
                    {
                        childBounds[i].m_end[this->bound_component_idx] = this->line.m[this->bound_component_idx];
                    }else
                    {
                        childBounds[i].m_start[this->bound_component_idx] = this->line.m[this->bound_component_idx];
                    }
                } else
                {
                    if(i == 0)
                    {
                        childBounds[i].d_bound_1 = this->d_bound;
                        childBounds[i].d_bound_2 = bounds.d_bound_1;
                    }else
                    {
                        childBounds[i].d_bound_1 = bounds.d_bound_1;
                        childBounds[i].d_bound_2 = -this->d_bound;
                    }
                }

                Eigen::Vector3f min_m;
                minimumDistances[i] = FindMinDist(query_point, childBounds[i].d_bound_1, childBounds[i].d_bound_2, childBounds[i].m_start, childBounds[i].m_end, min_m);
            }
        }

        // permutation = indices of children, from smallest to largest min dist
        std::array<uint8_t, 2> permutation = {0, 1};
        if(minimumDistances[0] > minimumDistances[1])
        {
            std::swap(permutation[0], permutation[1]);
        }

        unsigned int nbResultsFound = 0;
        auto resultsListLength = std::distance(out_first, out_last);
        for(uint8_t idx : permutation)
        {
            if(minimumDistances[idx] > max_dist)
            {
                break;
            }

            auto nbResultsInNode = FindNeighbours(query_point, out_first, out_last, max_dist, childBounds[idx]);
            nbResultsFound = std::min(nbResultsFound + nbResultsInNode, resultsListLength);
        }

        if((query_point.cross(line.d) - line.m).norm() < max_dist)
        {
            max_dist = insert(&line, out_first, out_last, query_point);
        }

        return nbResultsFound;
    }

    template<class OutputIt, typename = typename std::enable_if<std::is_same<Line, typename OutputIt::value_type>::value>::type>
    static float insert(const Line* elem, OutputIt out_first, OutputIt out_end, const Eigen::Vector3f& query_point)
    {
        auto distF = [](const Line* l, const Eigen::Vector3f& p)
        {
            return p.cross(l->d) - l->m;
        };

        auto it = std::lower_bound(out_first, out_end, elem, [&query_point, distF](const Line* c1, const Line* c2){
            auto dist1 = c1 == nullptr ? 1E99 : distF(c1, query_point).squaredNorm();
            auto dist2 = c2 == nullptr ? 1E99 : distF(c1, query_point).squaredNorm();
            return dist1 < dist2;
        });

        if(it != out_end)
        {
            iter_insert(it, out_end, elem);
        }

        auto lastElemPtr = *(out_end - 1);
        if(lastElemPtr == nullptr)
        {
            return std::numeric_limits<float>::max();
        }

        return distF(lastElemPtr, query_point).norm();
    }
};

class Tree
{
private:

public:
    std::array<TreeSector, 12> sectors;

    explicit Tree(std::array<TreeSector, 12> sectors) : sectors(std::move(sectors)) {}

    //void Add(const Line& line);
	//bool Remove(const Line* line);

    template<class OutputIt, typename = typename std::enable_if<std::is_same<Line, typename OutputIt::value_type>::value>::type>
    typename std::iterator_traits<OutputIt>::difference_type FindNeighbours(
    	const Eigen::Vector3f& query_point,
		OutputIt out_first, OutputIt out_last, 
    	float& max_dist
	) const
    {
        std::array<float, 12> minimumDistances {};
        for(int i = 0; i < minimumDistances.size(); ++i)
        {
            const TreeSector& sector = sectors[i];
            if(sector.rootNode == nullptr)
            {
                minimumDistances[i] = std::numeric_limits<float>::infinity();
            } else
            {
                Eigen::Vector3f min_m;
                minimumDistances[i] = FindMinDist(query_point, sector.bounds.d_bound_1, sector.bounds.d_bound_2, sector.bounds.m_start, sector.bounds.m_end, min_m);
            }
        }

        std::array<uint8_t, 12> permutation {};
        std::iota(permutation.begin(), permutation.end(), 0);
        std::sort(permutation.begin(), permutation.end(), [it = minimumDistances.begin()](uint8_t a, uint8_t b) {
            return *(it + a) < *(it + b);
        });

        //Search through sectors[idx].rootNode, ignoring sectors/bins with a mindist larger than searchRadius
        //Insert each line found into the output list, keeping the list sorted by distance.
        // Discard the last element, or don't insert if the new element is larger than all current results.
        //Set searchradius to the largest distance in the results list
        unsigned int nbResultsFound = 0;
        auto resultsListLength = std::distance(out_first, out_last);
        for(uint8_t idx : permutation)
        {
            if(minimumDistances[idx] > max_dist || sectors[idx].rootNode == nullptr)
            {
                break;
            }

            auto nbResultsInNode = sectors[idx].rootNode->FindNeighbours(
                    query_point, out_first, out_last, max_dist,
                    sectors[idx].bounds);
            nbResultsFound = std::min(nbResultsFound + nbResultsInNode, resultsListLength);
        }
        return nbResultsFound;
    }
};

class TreeBuilder
{
private:
    template<class LineIt, typename = typename std::enable_if<std::is_same<Line, typename LineIt::value_type>::value>::type>
    static std::unique_ptr<TreeNode> BuildNode(LineIt lines_begin, LineIt lines_end, const Bounds& bounds, int level)
    {
        auto lineCount = std::distance(lines_begin, lines_end);
        if(lineCount == 0)
        {
            return nullptr;
        }

        // Find axis with largest variance and split in 2 there
        NodeType type;
        uint8_t splitComponent = 0;
        LineIt pivot;
        Bounds subBounds1 = bounds;
        Bounds subBounds2 = bounds;

        // Calculate max moment variance
        Eigen::Array3f mVarianceVect = calc_vec3_variance(lines_begin, lines_end, [](const Line& l){return l.m; });
        auto mVariance = mVarianceVect.maxCoeff(&splitComponent);
        auto mBoundCompDist = (bounds.m_end[splitComponent] - bounds.m_start[splitComponent]);
        auto mMaxPossibleVariance = (mBoundCompDist * mBoundCompDist) / 4;
        auto mVarianceNormalized = mVariance / mMaxPossibleVariance;

        // Calculate directional variance
        // project direction vectors to bound domain, calculate variance of sine of angle to bound, and use this to decide NodeType
        /*Eigen::Vector3f cur_bound;
        Eigen::Vector3f bound_domain_normal;
        auto calc_sine = [](const Eigen::Vector3f& d, const Eigen::Vector3f& bound_domain_normal, const Eigen::Vector3f& cur_bound){
            Eigen::Vector3f cross1 = (d - bound_domain_normal * bound_domain_normal.dot(d)).normalized().cross(cur_bound);
            auto sin = cross1.norm();
            if(cross1.dot(bound_domain_normal) < 0)
            {
                sin *= -1;
            }
            return sin;
        };
        auto dVariance = calc_variance(lines_begin, lines_end, [&bound_domain_normal, &cur_bound, calc_sine](const Line& line){
            return calc_sine(line.d, bound_domain_normal, cur_bound);
        });
        auto dMaxPossibleVariance = ; //TODO
        auto dVarianceNormalized = dVariance / dMaxPossibleVariance;

        if(dVarianceNormalized > mVarianceNormalized)
        {
            type = NodeType::direction;

            // calculate new bound vector: calc cross product of dir vectors with cur bound vector to obtain sin,
            // sort by sin, take median, new bound vector is cross product of bound_domain_normal with median dir vect
            // Given bounds b1 and b2 in parent, and new bound vector nb, the childrens bounds are as follows:
            // child 1: {nb, b1}, child 2: {b1, -nb}

            std::sort(lines_begin, lines_end, [&cur_bound, &bound_domain_normal, calc_sine](const Line& l1, const Line& l2){
                auto sin1 = calc_sine(l1.d, bound_domain_normal, cur_bound);
                auto sin2 = calc_sine(l2.d, bound_domain_normal, cur_bound);
                return sin1 < sin2;
            });
            pivot = lines_begin + (lines_end - lines_begin)/2;
            Eigen::Vector3f dir_bound = bound_domain_normal.cross(pivot->d).normalized();
            subBounds1.d_bound_1;
            subBounds1.d_bound_2;
            subBounds2.d_bound_1;
            subBounds2.d_bound_2;
        } else*/
        {
            type = NodeType::moment;

            std::sort(lines_begin, lines_end, [splitComponent](const Line& l1, const Line& l2){
                return l1.m[splitComponent] < l2.m[splitComponent];
            });
            pivot = lines_begin + (lines_end - lines_begin)/2;
            subBounds1.m_end[splitComponent] = (*pivot).m[splitComponent];
            subBounds2.m_start[splitComponent] = (*pivot).m[splitComponent];
        }

        // Store iterators and bounds and then recurse
        auto node = std::make_unique<TreeNode>(type, splitComponent, *pivot);
        node->children[0] = BuildNode(lines_begin, pivot, subBounds1, level + 1);
        node->children[1] = BuildNode(pivot+1, lines_end, subBounds2, level + 1);
        return std::move(node);
    }

public:
	template<class LineIt, typename = typename std::enable_if<std::is_same<Line, typename LineIt::value_type>::value>::type>
    static Tree Build(LineIt lines_first, LineIt lines_last)
    {
        // Create sectors
        using Eigen::Vector3f;
        constexpr float max_dist = 100;
        std::array<TreeSector, 12> sectors {
            // Top sector
            TreeSector(Vector3f(1, 0, 0), Vector3f(1, 0, 0), Vector3f(-M_PI, 0, 0), Vector3f(M_PI, M_PI/4, max_dist)),
            TreeSector(Vector3f(-1, 0, 0), Vector3f(-1 ,0, 0), Vector3f(-M_PI, 0, 0), Vector3f(M_PI, M_PI/4, max_dist)),
            // Bottom sector
            TreeSector(Vector3f(1, 0, 0), Vector3f(1, 0, 0), Vector3f(-M_PI, 3*M_PI/4, 0), Vector3f(M_PI, M_PI, max_dist)),
            TreeSector(Vector3f(-1, 0, 0), Vector3f(-1, 0, 0), Vector3f(-M_PI, 3*M_PI/4, 0), Vector3f(M_PI, M_PI, max_dist)),
            // -X -Y sector
            TreeSector(Vector3f(0, 0, 1), Vector3f(0, 0, 1), Vector3f(-M_PI, M_PI/4, 0), Vector3f(-M_PI/2, 3*M_PI/4, max_dist)),
            TreeSector(Vector3f(0, 0, -1), Vector3f(0, 0, -1), Vector3f(-M_PI, M_PI/4, 0), Vector3f(-M_PI/2, 3*M_PI/4, max_dist)),
            // +X -Y sector
            TreeSector(Vector3f(0, 0, 1), Vector3f(0, 0, 1), Vector3f(-M_PI/2, M_PI/4, 0), Vector3f(0, 3*M_PI/4, max_dist)),
            TreeSector(Vector3f(0, 0, -1), Vector3f(0, 0, -1), Vector3f(-M_PI/2, M_PI/4, 0), Vector3f(0, 3*M_PI/4, max_dist)),
            // +X +Y sector
            TreeSector(Vector3f(0, 0, 1), Vector3f(0, 0, 1), Vector3f(0, M_PI/4, 0), Vector3f(M_PI/2, 3*M_PI/4, max_dist)),
            TreeSector(Vector3f(0, 0, -1), Vector3f(0, 0, -1), Vector3f(0, M_PI/4, 0), Vector3f(M_PI/2, 3*M_PI/4, max_dist)),
            // -X +Y sector
            TreeSector(Vector3f(0, 0, 1), Vector3f(0, 0, 1), Vector3f(M_PI/2, M_PI/4, 0), Vector3f(M_PI, 3*M_PI/4, max_dist)),
            TreeSector(Vector3f(0, 0, -1), Vector3f(0, 0, -1), Vector3f(M_PI/2, M_PI/4, 0), Vector3f(M_PI, 3*M_PI/4, max_dist))
        };

        // Sort lines into sectors
        std::array<LineIt, 12> line_sector_ends;
        {
            LineIt it_begin = lines_first;
            int idx = 0;
            for(TreeSector& sector : sectors)
            {
                for(LineIt it = it_begin; it < lines_last; ++it)
                {
                    const Line& line = *it;
                    if(Eigen::AlignedBox<float, 3>(sector.bounds.m_start, sector.bounds.m_end).contains(line.m)
                       && sector.bounds.d_bound_1.dot(line.d) >= 0 //greater or equal, or just greater?
                       && sector.bounds.d_bound_2.dot(line.d) >= 0)
                    {
                        std::iter_swap(it_begin, it);
                        it_begin++;
                    }
                }
                line_sector_ends[idx] = it_begin;
                idx++;
            }
        }

        // Subdivide sectors into bins
        {
            LineIt it_begin = lines_first;
            int idx = 0;
            for(TreeSector& sector : sectors)
            {
                LineIt it_end = line_sector_ends[idx];

                sector.rootNode = BuildNode(it_begin, it_end);

                it_begin = it_end;

                idx++;
            }
        }

        return Tree(std::move(sectors));
    }
};

}
