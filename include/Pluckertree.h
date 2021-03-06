#pragma once

#include <limits>
#include <cstdint>
#include <numeric>
#include <memory>
#include <Eigen/Dense>
#include "MathUtil.h"
#include <random>
#include <iostream>
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

double FindMinHitDist(
        const Eigen::Vector3f& point,
        const Eigen::Vector3f& point_normal,
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

void show_me_the_grid(std::string& file,
                      const Eigen::Vector3f& dlb,
                      const Eigen::Vector3f& dub,
                      const Eigen::Vector3f& mlb,
                      const Eigen::Vector3f& mub,
                      const Eigen::Vector3f& q);

class Line
{
public:
	Line(Eigen::Vector3f d, Eigen::Vector3f m) : d(std::move(d)), m(std::move(m)) {};

	static Line FromPointAndDirection(const Eigen::Vector3f& point, const Eigen::Vector3f& direction)
	{
	    auto p = point - point.dot(direction)*direction;
	    auto m = p.cross(direction);
	    return Line(direction, m);

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

    Eigen::Vector3f d; // Carthesian
	Eigen::Vector3f m; // Carthesian

    bool operator==(const Line& rhs) const
    {
        return (d == rhs.d) && (m == rhs.m);
    }
    bool operator!=(const Line& rhs) const
    {
        return !operator==(rhs);
    }
};

template<class Content, Line Content::*line_member>
class TreeNode;

struct Bounds
{
    Eigen::Vector3f d_bound_1; // Carthesian
    Eigen::Vector3f d_bound_2; // Carthesian
    Eigen::Vector3f m_start; // Spherical
    Eigen::Vector3f m_end; // Spherical

    Bounds(Eigen::Vector3f d_bound_1, Eigen::Vector3f d_bound_2, Eigen::Vector3f m_start, Eigen::Vector3f m_end)
    : d_bound_1(std::move(d_bound_1)), d_bound_2(std::move(d_bound_2)), m_start(std::move(m_start)), m_end(std::move(m_end)) {}

    Bounds() = default;

    void Clip(Eigen::Vector3f& moment) const
    {
        // Technically, we should account for -PI = PI in azimuth, but due to choice of sectors this method is fine.
        moment.x() = std::max(m_start.x(), std::min(m_end.x(), moment.x()));
        moment.y() = std::max(m_start.y(), std::min(m_end.y(), moment.y()));
        moment.z() = std::max(m_start.z(), std::min(m_end.z(), moment.z()));
    }

    bool ContainsMoment(const Eigen::Vector3f& moment /*spher coords*/) const
    {
        return Eigen::AlignedBox<float, 3>(m_start, m_end).contains(moment);
    }
};

template<class Content, Line Content::*line_member>
class TreeSector
{
public:
    Bounds bounds;
    std::unique_ptr<TreeNode<Content, line_member>> rootNode;

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
 *
 *  This function will insert a copy of the given value before the specified location.
 *  All values will shift towards the end, with the last element being discarded.
 *  position must be smaller than end.
 */
template<typename Iterator, typename Value, typename = typename std::enable_if<std::is_same<Value, typename std::iterator_traits<Iterator>::value_type>::value>::type>
void iter_insert(Iterator position, Iterator end, const Value& val)
{
    assert(position < end);

    for(Iterator it = end-2; it >= position; --it)
    {
        *(it+1) = *it;
    }
    *position = val;
}

constexpr float margin = 0;
struct Diag {
    static thread_local unsigned int visited;
    static thread_local unsigned int minimizations;
    static std::optional<std::function<void(float best_so_far_dist, float cur_line_dist, float cur_node_mindist, int cur_depth)>> on_node_visited;
    static std::optional<std::function<void(float best_so_far_dist, float cur_node_mindist, int cur_depth)>> on_node_enter;
    static std::optional<std::function<void(float best_so_far_dist, float cur_line_dist, float cur_node_mindist, int cur_depth)>> on_node_leave;
    static std::optional<std::function<void(float m_phi_var, float m_gamma_var, float m_radius_var, float d_var)>> on_build_variance_calculated;
    static bool force_visit_all;

    static void reset()
    {
        visited = 0;
        minimizations = 0;
        on_node_visited = std::nullopt;
        on_node_enter = std::nullopt;
        on_node_leave = std::nullopt;
        on_build_variance_calculated = std::nullopt;
        force_visit_all = false;
    }
};

template<class Content, class DistF, class OutputIt, typename = typename std::enable_if<std::is_same<const Content*, typename std::iterator_traits<OutputIt>::value_type>::value>::type>
static float insert(const Content* elem, OutputIt out_first, OutputIt out_end, const Eigen::Vector3f& query_point, const DistF& distF)
{
    auto it = std::lower_bound(out_first, out_end, elem, [&query_point, distF](const Content* c1, const Content* c2){
        auto dist1 = c1 == nullptr ? std::numeric_limits<float>::infinity() : distF(c1, query_point).squaredNorm();
        auto dist2 = c2 == nullptr ? std::numeric_limits<float>::infinity() : distF(c2, query_point).squaredNorm();
        return dist1 < dist2;
    });

    if(it != out_end)
    {
        iter_insert(it, out_end, elem);
    }

    auto lastElemPtr = *(out_end - 1);
    if(lastElemPtr == nullptr)
    {
        return std::numeric_limits<float>::infinity();
    }

    return distF(lastElemPtr, query_point).norm();
}

template<class Content, Line Content::*line_member>
class TreeNode
{
public:
    std::array<std::unique_ptr<TreeNode<Content, line_member>>, 2> children;
    Content content;
    Eigen::Vector3f d_bound; //carthesian
    float m_component; //spherical
    NodeType type : 1;
    //uint8_t bound_idx : 1;
    uint8_t bound_component_idx : 2;
    //uint8_t pad1 : 6;
    //uint8_t pad2[7];

    TreeNode(uint8_t bound_component_idx, float m_component, Content content)
        : type(NodeType::moment), bound_component_idx(bound_component_idx), m_component(m_component), content(std::move(content)), children() {}

    TreeNode(Eigen::Vector3f d_bound, Content content)
            : type(NodeType::direction), d_bound(std::move(d_bound)), content(std::move(content)), children() {}


    template<class OutputIt, typename = typename std::enable_if<std::is_same<const Content*, typename std::iterator_traits<OutputIt>::value_type>::value>::type>
    typename std::iterator_traits<OutputIt>::difference_type FindNeighbours(
            const Eigen::Vector3f& query_point,
            OutputIt out_first, OutputIt out_last,
            float& max_dist,
            const Bounds& bounds,
            const Eigen::Vector3f& moment_min_hint,
            int depth,
            float moment_min_hint_dist
    ) const
    {
        if(Diag::on_node_enter.has_value())
        {
            (*Diag::on_node_enter)(max_dist, moment_min_hint_dist, depth);
        }

        Diag::visited++;
        std::array<float, 2> minimumDistances {};
        std::array<Bounds, 2> childBounds {};
        std::array<Eigen::Vector3f, 2> childMomentMinima {};

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
                        childBounds[0].m_end[this->bound_component_idx] = m_component;
                    }else
                    {
                        childBounds[1].m_start[this->bound_component_idx] = m_component;
                    }

                    if(childBounds[i].ContainsMoment(moment_min_hint)) //TODO: test
                    {
                        childMomentMinima[i] = moment_min_hint;
                        minimumDistances[i] = moment_min_hint_dist;
                        continue;
                    }
                } else
                {
                    if(i == 0)
                    {
                        childBounds[i].d_bound_1 = this->d_bound;
                        childBounds[i].d_bound_2 = bounds.d_bound_2;
                    }else
                    {
                        childBounds[i].d_bound_1 = bounds.d_bound_1;
                        childBounds[i].d_bound_2 = -this->d_bound;
                    }
                }

                childMomentMinima[i] = moment_min_hint;
                childBounds[i].Clip(childMomentMinima[i]);
                minimumDistances[i] = FindMinDist(query_point, childBounds[i].d_bound_1, childBounds[i].d_bound_2, childBounds[i].m_start, childBounds[i].m_end, childMomentMinima[i]);
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
            if((minimumDistances[idx] > max_dist+margin && !Diag::force_visit_all) || children[idx] == nullptr) //max_dist_check
            {
                break;
            }

            auto nbResultsInNode = children[idx]->FindNeighbours(query_point, out_first, out_last, max_dist, childBounds[idx], childMomentMinima[idx], depth+1, minimumDistances[idx]);
            nbResultsFound = std::min(nbResultsFound + nbResultsInNode, resultsListLength);
        }

        auto distF = [](const Content* c, const Eigen::Vector3f& p)
        {
            const auto& l = c->*line_member;
            Eigen::Vector3f v = p.cross(l.d) - l.m;
            return v;
        };

        auto dist = distF(&content, query_point).norm();

        if(Diag::on_node_visited.has_value())
        {
            (*Diag::on_node_visited)(max_dist, dist, moment_min_hint_dist, depth);
        }

        if(dist < max_dist+margin) //max_dist_check
        {
            max_dist = insert(&content, out_first, out_last, query_point, distF);
            nbResultsFound++;
        }

        if(Diag::on_node_leave.has_value())
        {
            (*Diag::on_node_leave)(max_dist, dist, moment_min_hint_dist, depth);
        }

        return nbResultsFound;
    }

    template<class OutputIt, typename = typename std::enable_if<std::is_same<const Content*, typename std::iterator_traits<OutputIt>::value_type>::value>::type>
    typename std::iterator_traits<OutputIt>::difference_type FindNearestHits(
            const Eigen::Vector3f& query_point,
            const Eigen::Vector3f& query_normal,
            OutputIt out_first, OutputIt out_last,
            float& max_dist,
            const Bounds& bounds,
            const Eigen::Vector3f& moment_min_hint,
            int depth,
            float moment_min_hint_dist
    ) const
    {
        Diag::visited++;
        std::array<float, 2> minimumDistances {};
        std::array<Bounds, 2> childBounds {};
        std::array<Eigen::Vector3f, 2> childMomentMinima {};

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
                        childBounds[0].m_end[this->bound_component_idx] = m_component;
                    }else
                    {
                        childBounds[1].m_start[this->bound_component_idx] = m_component;
                    }

                    if(childBounds[i].ContainsMoment(moment_min_hint)) //TODO: test
                    {
                        childMomentMinima[i] = moment_min_hint;
                        minimumDistances[i] = moment_min_hint_dist;
                        continue;
                    }
                } else
                {
                    if(i == 0)
                    {
                        childBounds[i].d_bound_1 = this->d_bound;
                        childBounds[i].d_bound_2 = bounds.d_bound_2;
                    }else
                    {
                        childBounds[i].d_bound_1 = bounds.d_bound_1;
                        childBounds[i].d_bound_2 = -this->d_bound;
                    }
                }

                childMomentMinima[i] = moment_min_hint;
                childBounds[i].Clip(childMomentMinima[i]);
                minimumDistances[i] = FindMinHitDist(query_point, query_normal, childBounds[i].d_bound_1, childBounds[i].d_bound_2, childBounds[i].m_start, childBounds[i].m_end, childMomentMinima[i]);
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
            if((minimumDistances[idx] > max_dist+margin && !Diag::force_visit_all) || children[idx] == nullptr) //max_dist_check
            {
                break;
            }

            auto nbResultsInNode = children[idx]->FindNearestHits(query_point, query_normal, out_first, out_last, max_dist, childBounds[idx], childMomentMinima[idx], depth+1, minimumDistances[idx]);
            nbResultsFound = std::min(nbResultsFound + nbResultsInNode, resultsListLength);
        }

        //auto dist = (query_point.cross(line.d) - line.m).norm();
        auto distF = [&query_normal](const Content* c, const Eigen::Vector3f& p){
            const auto& l = c->*line_member;
            Eigen::Vector3f l0 = (l.d.cross(l.m));
            Eigen::Vector3f intersection = l0 + (l.d * (p - l0).dot(query_normal)/(l.d.dot(query_normal)));
            Eigen::Vector3f vect = intersection - p;
            return vect;
        };
        auto dist = distF(&content, query_point).norm();
        if(Diag::on_node_visited.has_value())
        {
            (*Diag::on_node_visited)(max_dist, dist, moment_min_hint_dist, depth);
        }
        if(dist < max_dist+margin) //max_dist_check
        {
            max_dist = insert(&content, out_first, out_last, query_point, distF);
            nbResultsFound++;
        }

        return nbResultsFound;
    }
};


template<typename Content, Line Content::*line_member>
class Tree
{
private:
    size_t _size;

public:
    std::array<TreeSector<Content, line_member>, 32> sectors;


    explicit Tree(size_t size, std::array<TreeSector<Content, line_member>, 32> sectors) : _size(size), sectors(std::move(sectors)) {}

    size_t size() const { return _size; };

    //void Add(const Line& line);
	//bool Remove(const Line* line);

	template<class OutputIt, typename = typename std::enable_if<std::is_same<const Content*, typename std::iterator_traits<OutputIt>::value_type>::value>::type>
    typename std::iterator_traits<OutputIt>::difference_type FindNeighbours(
    	const Eigen::Vector3f& query_point,
		OutputIt out_first, OutputIt out_last, 
    	float& max_dist
	) const
    {
        Diag::visited = 0;
        Diag::minimizations = 0;

        std::array<float, 32> minimumDistances {};
        std::array<Eigen::Vector3f, 32> moment_min_hints {};
        for(int i = 0; i < minimumDistances.size(); ++i)
        {
            const auto& sector = sectors[i];
            if(sector.rootNode == nullptr)
            {
                minimumDistances[i] = std::numeric_limits<float>::infinity();
            } else
            {
                moment_min_hints[i] = sector.bounds.m_start + (sector.bounds.m_end - sector.bounds.m_start)/2;
                minimumDistances[i] = FindMinDist(query_point, sector.bounds.d_bound_1, sector.bounds.d_bound_2, sector.bounds.m_start, sector.bounds.m_end, moment_min_hints[i]);
            }
        }

        std::array<uint8_t, 32> permutation {};
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
            if((minimumDistances[idx] > max_dist+margin && !Diag::force_visit_all) || sectors[idx].rootNode == nullptr) //max_dist_check
            {
                break;
            }

            const auto& curBounds = sectors[idx].bounds;
            auto nbResultsInNode = sectors[idx].rootNode->FindNeighbours(
                    query_point, out_first, out_last, max_dist, curBounds, moment_min_hints[idx], 0, minimumDistances[idx]);
            nbResultsFound = std::min(nbResultsFound + nbResultsInNode, resultsListLength);
        }
        //std::cout << "visited " << Diag::visited << "/" << size() << std::endl;
        return nbResultsFound;
    }

    template<class OutputIt, typename = typename std::enable_if<std::is_same<const Content*, typename std::iterator_traits<OutputIt>::value_type>::value>::type>
    typename std::iterator_traits<OutputIt>::difference_type FindNearestHits(
            const Eigen::Vector3f& query_point,
            const Eigen::Vector3f& query_normal,
            OutputIt out_first, OutputIt out_last,
            float& max_dist
    ) const
    {
        Diag::visited = 0;
        Diag::minimizations = 0;

        std::array<float, 32> minimumDistances {};
        std::array<Eigen::Vector3f, 32> moment_min_hints {};
        for(int i = 0; i < minimumDistances.size(); ++i)
        {
            const auto& sector = sectors[i];
            if(sector.rootNode == nullptr)
            {
                minimumDistances[i] = std::numeric_limits<float>::infinity();
            } else
            {
                moment_min_hints[i] = sector.bounds.m_start + (sector.bounds.m_end - sector.bounds.m_start)/2;
                minimumDistances[i] = FindMinHitDist(query_point, query_normal, sector.bounds.d_bound_1, sector.bounds.d_bound_2, sector.bounds.m_start, sector.bounds.m_end, moment_min_hints[i]);
            }
        }

        std::array<uint8_t, 32> permutation {};
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
            if((minimumDistances[idx] > max_dist+margin && !Diag::force_visit_all) || sectors[idx].rootNode == nullptr) //max_dist_check
            {
                break;
            }

            const auto& curBounds = sectors[idx].bounds;
            auto nbResultsInNode = sectors[idx].rootNode->FindNearestHits(
                    query_point, query_normal, out_first, out_last, max_dist, curBounds, moment_min_hints[idx], 0, minimumDistances[idx]);
            nbResultsFound = std::min(nbResultsFound + nbResultsInNode, resultsListLength);
        }
        //std::cout << "visited " << Diag::visited << "/" << size() << std::endl;
        return nbResultsFound;
    }
};

struct MyRand{
    static std::random_device rand_dev;
};

template<class Content, Line Content::*line_member>
class TreeBuilder
{
    using Node = TreeNode<Content, line_member>;
    using Sector = TreeSector<Content, line_member>;

private:
    template<class LineIt, typename = typename std::enable_if<std::is_same<Content, typename std::iterator_traits<LineIt>::value_type>::value>::type>
    static std::unique_ptr<Node> BuildNode(LineIt lines_begin, LineIt lines_end, const Bounds& bounds, int level)
    {
        auto lineCount = std::distance(lines_begin, lines_end);
        if(lineCount == 0)
        {
            return nullptr;
        }

        // Find axis with largest variance and split in 2 there
        std::unique_ptr<Node> node;
        //NodeType type;
        uint8_t splitComponent = 0;
        Bounds subBounds1 = bounds;
        Bounds subBounds2 = bounds;
        LineIt pivot;
        std::uniform_real_distribution<float> unit{0.0f, 1.0f};
        float v = unit(MyRand::rand_dev);

        ///////////////////////////////
        // Calculate directional variance
        // project direction vectors to bound domain, calculate variance of sine of angle to bound, and use this to decide NodeType
        const Eigen::Vector3f& cur_bound = bounds.d_bound_1;
        Eigen::Vector3f bound_domain_normal = bounds.d_bound_1.cross(bounds.d_bound_2).normalized();
        if(bound_domain_normal.dot(bounds.m_start) < 0)
        {
            bound_domain_normal *= -1;
        }

        auto calc_sine = [](const Eigen::Vector3f& d, const Eigen::Vector3f& bound_domain_normal, const Eigen::Vector3f& cur_bound){
            Eigen::Vector3f cross1 = (d - bound_domain_normal * bound_domain_normal.dot(d)).normalized().cross(cur_bound);
            auto sin = cross1.norm();
            if(cross1.dot(bound_domain_normal) < 0)
            {
                sin *= -1;
            }
            return sin;
        };
        /*auto dVariance = calc_pop_variance(lines_begin, lines_end, [&bound_domain_normal, &cur_bound, calc_sine](const Content& c){
            const auto& line = c.*line_member;
            return std::asin(calc_sine(line.d, bound_domain_normal, cur_bound));
        });
        auto dMaxSin = bounds.d_bound_1.cross(bounds.d_bound_2).norm();
        //auto dMaxPossibleVariance = (dMaxSin * dMaxSin)/4; //Assumes angle between bounds >= 90°
        //auto dVarianceNormalized = dVariance / dMaxPossibleVariance;
        auto dVarianceNormFactor = std::asin(dMaxSin);
        dVarianceNormFactor = dVarianceNormFactor * dVarianceNormFactor;
        auto dVarianceNormalized = dVariance;
        */
        ///////////////////////////////

        bool splitOnDir = false;
        ///////////////////////////////
        // Calculate max moment variance
#define M_LEVEL_ALTERNATING
#if defined(M_MAX_VAR) || defined(M_MIN_VAR)
        #undef M_MAX_VAR
        Eigen::Array3f mVarianceVect = calc_vec3_pop_variance(lines_begin, lines_end, [](const Content& c){return cart2spherical((c.*line_member).m); });
        Eigen::Array3f mVarianceNormFact = (bounds.m_end - bounds.m_start).array();
        mVarianceNormFact = mVarianceNormFact * mVarianceNormFact;
        mVarianceVect = mVarianceVect.cwiseQuotient(mVarianceNormFact);
    #if defined(M_MAX_VAR)
        auto mVariance = mVarianceVect.maxCoeff(&splitComponent);
    #elif defined(M_MIN_VAR)
        auto mVariance = mVarianceVect.minCoeff(&splitComponent);
    #endif
        //splitOnDir = dVarianceNormalized > mVariance;

        /*if(level < 5 && level % 2 == 0)
        {
            splitComponent = 2;
        }*/

        splitOnDir = v < 0.25;
        unsigned int pivot_i = lineCount/2;
#elif defined(M_MAX_MAD) || defined(M_MIN_MAD)
        #undef M_MAX_MAD
        float phi_mad = calc_MAD(lines_begin, lines_end, [](const Content& c){return cart2spherical((c.*line_member).m)[0]; });
        float theta_mad = calc_MAD(lines_begin, lines_end, [](const Content& c){return cart2spherical((c.*line_member).m)[1]; });
        float radius_mad = calc_MAD(lines_begin, lines_end, [](const Content& c){return cart2spherical((c.*line_member).m)[2]; });
        Eigen::Array3f mMADVect {phi_mad, theta_mad, radius_mad};
        Eigen::Array3f mMADNormFact = (bounds.m_end - bounds.m_start).array();
        //mMADNormFact[2] = mMADNormFact[2] / 3.0f;
        //std::cout << mMADVect.transpose() << " | " << (mMADVect/mMADNormFact).transpose() << std::endl;
        mMADVect = mMADVect.cwiseQuotient(mMADNormFact);
    #if defined(M_MAX_MAD)
        auto mMAD = mMADVect.maxCoeff(&splitComponent);
    #elif defined(M_MIN_MAD)
        auto mMAD = mMADVect.minCoeff(&splitComponent);
    #endif
        /*if(level == 0)
        {
            splitComponent = 2;
        }else if(level == 1)
        {
            splitComponent = 0;
        }*/

        //splitOnDir = dVarianceNormalized > mVariance;
        splitOnDir = v < 0.25;
        unsigned int pivot_i = lineCount/2;
#elif defined(M_RANDOM)
        if(v < 0.37)
        {
            splitComponent = 0;
        }else if(v < 0.65)
        {
            splitComponent = 1;
        }else{
            splitComponent = 2;
        }
        splitOnDir = v < 0.25;
        unsigned int pivot_i = lineCount/2;
#elif defined(M_LEVEL_ALTERNATING)
        splitComponent = level % 4;
        splitOnDir = (level % 4) == 3;
        unsigned int pivot_i = lineCount/2;
#elif defined(M_VOLUME_HEURISTIC)
        unsigned int cur_best_dim_i = 0;
        unsigned int cur_best_i = 0;
        float cur_best_h = 1E99;

        for(int dim_i = 0; dim_i < 3; ++dim_i)
        {
            std::sort(lines_begin, lines_end, [splitComponent](const Content& c1, const Content& c2){
                const auto& l1 = c1.*line_member;
                const auto& l2 = c2.*line_member;
                return cart2spherical(l1.m)[splitComponent] < cart2spherical(l2.m)[splitComponent];
            });

            for(unsigned int i = 1; i < lineCount-1; ++i)
            {
                const Line& l1 = (*(lines_begin+i-1)).*line_member;
                const Line& l2 = (*(lines_begin+i+1)).*line_member;
                subBounds1.m_end[dim_i] = cart2spherical(l1.m)[dim_i];
                subBounds2.m_start[dim_i] = cart2spherical(l2.m)[dim_i];

                float vol1;
                {
                    float r1 = subBounds1.m_start.z();
                    float r2 = subBounds1.m_end.z();
                    vol1 = (((r2*r2*r2) - (r1*r1*r1))/2.0f) * (-std::cos(subBounds1.m_end.y()) + std::cos(subBounds1.m_start.y())) * (subBounds1.m_end.x() - subBounds1.m_start.x());
                }

                float vol2;
                {
                    float r1 = subBounds2.m_start.z();
                    float r2 = subBounds2.m_end.z();
                    vol2 = (((r2*r2*r2) - (r1*r1*r1))/2.0f) * (-std::cos(subBounds2.m_end.y()) + std::cos(subBounds2.m_start.y())) * (subBounds2.m_end.x() - subBounds2.m_start.x());
                }

                auto node1ElemCount = i;
                auto node2ElemCount = (lineCount - i - 1);

                //float heuristic = (vol1 * node1ElemCount) + (vol2 * node2ElemCount);
                //float heuristic = (vol1) + (vol2);
                //float heuristic = (vol1/node1ElemCount) + (vol2/node2ElemCount);
                float heuristic = (vol1*node1ElemCount) + (vol2*node2ElemCount);

                if(heuristic < cur_best_h)
                {
                    cur_best_h = heuristic;
                    cur_best_dim_i = dim_i;
                    cur_best_i = i;
                }
            }
            subBounds1.m_end[dim_i] = bounds.m_end[dim_i];
            subBounds2.m_start[dim_i] = bounds.m_start[dim_i];
        }
        splitComponent = cur_best_dim_i;
        unsigned int pivot_i = cur_best_i;
        splitOnDir = v < 0.25;
#elif defined(M_LONGEST_DIST)
        unsigned int cur_best_dim_i = 0;
        unsigned int cur_best_i = 0;
        float cur_best_h = 0;

        for(int dim_i = 0; dim_i < 3; ++dim_i)
        {
            std::sort(lines_begin, lines_end, [splitComponent](const Content& c1, const Content& c2){
                const auto& l1 = c1.*line_member;
                const auto& l2 = c2.*line_member;
                return cart2spherical(l1.m)[splitComponent] < cart2spherical(l2.m)[splitComponent];
            });

            for(unsigned int i = 1; i < lineCount-1; ++i)
            {
                const Line& l1 = (*(lines_begin+i-1)).*line_member;
                const Line& l2 = (*(lines_begin+i+1)).*line_member;

                float heuristic = ((cart2spherical(l2.m)[dim_i]) - (cart2spherical(l1.m)[dim_i]))/(bounds.m_end[dim_i] - bounds.m_start[dim_i]);

                if(heuristic > cur_best_h)
                {
                    cur_best_h = heuristic;
                    cur_best_dim_i = dim_i;
                    cur_best_i = i;
                }
            }
        }
        splitComponent = cur_best_dim_i;
        unsigned int pivot_i = cur_best_i;
        splitOnDir = v < 0.25;
#elif defined(M_PACK_AND_CUT)
        bool isPackNode = level % 4 == 0;
        unsigned int cur_best_dim_i = 0;
        unsigned int cur_best_i = 0;
        float cur_best_h = 0;
        std::array<float,3> cur_best_hs {0,0,0};

        for(int dim_i = 0; dim_i < 3; ++dim_i)
        {
            std::sort(lines_begin, lines_end, [splitComponent](const Content& c1, const Content& c2){
                const auto& l1 = c1.*line_member;
                const auto& l2 = c2.*line_member;
                return cart2spherical(l1.m)[splitComponent] < cart2spherical(l2.m)[splitComponent];
            });

            for(unsigned int i = 1; i < lineCount-1; ++i)
            {
                const Line& l1 = (*(lines_begin+i-1)).*line_member;
                const Line& l2 = (*(lines_begin+i+1)).*line_member;
                subBounds1.m_end[dim_i] = cart2spherical(l1.m)[dim_i];
                subBounds2.m_start[dim_i] = cart2spherical(l2.m)[dim_i];

                float vol1;
                {
                    float r1 = subBounds1.m_start.z();
                    float r2 = subBounds1.m_end.z();
                    vol1 = (((r2*r2*r2) - (r1*r1*r1))/2.0f) * (-std::cos(subBounds1.m_end.y()) + std::cos(subBounds1.m_start.y())) * (subBounds1.m_end.x() - subBounds1.m_start.x());
                }

                float vol2;
                {
                    float r1 = subBounds2.m_start.z();
                    float r2 = subBounds2.m_end.z();
                    vol2 = (((r2*r2*r2) - (r1*r1*r1))/2.0f) * (-std::cos(subBounds2.m_end.y()) + std::cos(subBounds2.m_start.y())) * (subBounds2.m_end.x() - subBounds2.m_start.x());
                }

                auto node1ElemCount = i;
                auto node2ElemCount = (lineCount - i - 1);

                /*float heuristic1 = 0.5*(vol1/node1ElemCount) + (node2ElemCount/vol2); //cut, pack
                float heuristic2 = (node1ElemCount/vol1) + 0.5*(vol2/node2ElemCount); //pack, cut
                float heuristic = std::max(heuristic1, heuristic2);*/

                float heuristic;
                if(isPackNode)
                {
                    heuristic = std::max(vol1/node1ElemCount, vol2/node2ElemCount);
                }else{
                    heuristic = std::max(node1ElemCount/vol1, node2ElemCount/vol2);
                }

                if(heuristic > cur_best_hs[dim_i]) { cur_best_hs[dim_i] = heuristic; }

                if(heuristic > cur_best_h)
                {
                    cur_best_h = heuristic;
                    cur_best_dim_i = dim_i;
                    cur_best_i = i;
                }
            }
            subBounds1.m_end[dim_i] = bounds.m_end[dim_i];
            subBounds2.m_start[dim_i] = bounds.m_start[dim_i];
        }
        splitComponent = cur_best_dim_i;
        unsigned int pivot_i = cur_best_i;
        splitOnDir = v < 0.25;
#endif

        ///////////////////////////////

        if(Diag::on_build_variance_calculated.has_value())
        {
            //Diag::on_build_variance_calculated.value()(mVarianceVect.x(), mVarianceVect.y(), mVarianceVect.z(), 0/*dVarianceNormalized*/);
            //Diag::on_build_variance_calculated.value()(cur_best_hs[0], cur_best_hs[1], cur_best_hs[2], 0/*dVarianceNormalized*/);
        }

        if(splitOnDir)
        {
            // calculate new bound vector: calc cross product of dir vectors with cur bound vector to obtain sin,
            // sort by sin, take median, new bound vector is cross product of bound_domain_normal with median dir vect
            // Given bounds b1 and b2 in parent, and new bound vector nb, the childrens bounds are as follows:
            // child 1: {nb, b2}, child 2: {b1, -nb}

            std::sort(lines_begin, lines_end, [&cur_bound, &bound_domain_normal, calc_sine](const Content& c1, const Content& c2){
                const auto& l1 = c1.*line_member;
                const auto& l2 = c2.*line_member;
                auto sin1 = calc_sine(l1.d, bound_domain_normal, cur_bound);
                auto sin2 = calc_sine(l2.d, bound_domain_normal, cur_bound);
                return sin1 < sin2;
            });
            pivot = lines_begin + (lines_end - lines_begin)/2;
            const auto& pivotLine = (*pivot).*line_member;
            Eigen::Vector3f dir_bound = bound_domain_normal.cross(pivotLine.d).normalized();
            subBounds1.d_bound_1 = dir_bound;
            subBounds1.d_bound_2 = bounds.d_bound_2;
            subBounds2.d_bound_1 = bounds.d_bound_1;
            subBounds2.d_bound_2 = -dir_bound;

            node = std::make_unique<Node>(dir_bound, *pivot);
        } else
        {
            std::sort(lines_begin, lines_end, [splitComponent](const Content& c1, const Content& c2){
                const auto& l1 = c1.*line_member;
                const auto& l2 = c2.*line_member;
                return cart2spherical(l1.m)[splitComponent] < cart2spherical(l2.m)[splitComponent];
            });
            pivot = lines_begin + pivot_i;

            const auto& pivotLine = (*pivot).*line_member;
            auto splitCompVal = cart2spherical(pivotLine.m)[splitComponent];
            subBounds1.m_end[splitComponent] = splitCompVal;
            subBounds2.m_start[splitComponent] = splitCompVal;

            node = std::make_unique<Node>(splitComponent, splitCompVal, *pivot);
        }

        // Store iterators and bounds and then recurse
        node->children[0] = BuildNode(lines_begin, pivot, subBounds1, level + 1);
        node->children[1] = BuildNode(pivot+1, lines_end, subBounds2, level + 1);
        return std::move(node);
    }

public:
    template<class LineIt, typename = typename std::enable_if<std::is_same<Content, typename std::iterator_traits<LineIt>::value_type>::value>::type>
    static Tree<Content, line_member> Build(LineIt lines_first, LineIt lines_last)
    {
        // Create sectors
        using Eigen::Vector3f;
        constexpr float max_dist = 150; //TODO
        constexpr float min_dist = 1e-3;
        std::array<Sector, 32> sectors {
            // Top sector
            Sector(Vector3f(1, 0, 0), Vector3f(0, 1, 0), Vector3f(-M_PI, 0, min_dist), Vector3f(0, M_PI/4, max_dist)),
            Sector(Vector3f(0, 1, 0), Vector3f(-1, 0, 0), Vector3f(-M_PI, 0, min_dist), Vector3f(0, M_PI/4, max_dist)),
            Sector(Vector3f(-1, 0, 0), Vector3f(0, -1, 0), Vector3f(-M_PI, 0, min_dist), Vector3f(0, M_PI/4, max_dist)),
            Sector(Vector3f(0, -1, 0), Vector3f(1, 0, 0), Vector3f(-M_PI, 0, min_dist), Vector3f(0, M_PI/4, max_dist)),
            Sector(Vector3f(1, 0, 0), Vector3f(0, 1, 0), Vector3f(0, 0, min_dist), Vector3f(M_PI, M_PI/4, max_dist)),
            Sector(Vector3f(0, 1, 0), Vector3f(-1, 0, 0), Vector3f(0, 0, min_dist), Vector3f(M_PI, M_PI/4, max_dist)),
            Sector(Vector3f(-1, 0, 0), Vector3f(0, -1, 0), Vector3f(0, 0, min_dist), Vector3f(M_PI, M_PI/4, max_dist)),
            Sector(Vector3f(0, -1, 0), Vector3f(1, 0, 0), Vector3f(0, 0, min_dist), Vector3f(M_PI, M_PI/4, max_dist)),
            // Bottom sector
            Sector(Vector3f(1, 0, 0), Vector3f(0, 1, 0), Vector3f(-M_PI, 3*M_PI/4, min_dist), Vector3f(0, M_PI, max_dist)),
            Sector(Vector3f(0, 1, 0), Vector3f(-1, 0, 0), Vector3f(-M_PI, 3*M_PI/4, min_dist), Vector3f(0, M_PI, max_dist)),
            Sector(Vector3f(-1, 0, 0), Vector3f(0, -1, 0), Vector3f(-M_PI, 3*M_PI/4, min_dist), Vector3f(0, M_PI, max_dist)),
            Sector(Vector3f(0, -1, 0), Vector3f(1, 0, 0), Vector3f(-M_PI, 3*M_PI/4, min_dist), Vector3f(0, M_PI, max_dist)),
            Sector(Vector3f(1, 0, 0), Vector3f(0, 1, 0), Vector3f(0, 3*M_PI/4, min_dist), Vector3f(M_PI, M_PI, max_dist)),
            Sector(Vector3f(0, 1, 0), Vector3f(-1, 0, 0), Vector3f(0, 3*M_PI/4, min_dist), Vector3f(M_PI, M_PI, max_dist)),
            Sector(Vector3f(-1, 0, 0), Vector3f(0, -1, 0), Vector3f(0, 3*M_PI/4, min_dist), Vector3f(M_PI, M_PI, max_dist)),
            Sector(Vector3f(0, -1, 0), Vector3f(1, 0, 0), Vector3f(0, 3*M_PI/4, min_dist), Vector3f(M_PI, M_PI, max_dist)),
            // -X -Y sector
            Sector(Vector3f(0, 0, 1), Vector3f(1, -1, 0).normalized(), Vector3f(-M_PI, M_PI/4, min_dist), Vector3f(-M_PI/2, 3*M_PI/4, max_dist)),
            Sector(Vector3f(1, -1, 0).normalized(), Vector3f(0, 0, -1), Vector3f(-M_PI, M_PI/4, min_dist), Vector3f(-M_PI/2, 3*M_PI/4, max_dist)),
            Sector(Vector3f(0, 0, -1), Vector3f(-1, 1, 0).normalized(), Vector3f(-M_PI, M_PI/4, min_dist), Vector3f(-M_PI/2, 3*M_PI/4, max_dist)),
            Sector(Vector3f(-1, 1, 0).normalized(), Vector3f(0, 0, 1), Vector3f(-M_PI, M_PI/4, min_dist), Vector3f(-M_PI/2, 3*M_PI/4, max_dist)),
            // +X -Y sector
            Sector(Vector3f(0, 0, 1), Vector3f(-1, -1, 0).normalized(), Vector3f(-M_PI/2, M_PI/4, min_dist), Vector3f(0, 3*M_PI/4, max_dist)),
            Sector(Vector3f(-1, -1, 0).normalized(), Vector3f(0, 0, -1), Vector3f(-M_PI/2, M_PI/4, min_dist), Vector3f(0, 3*M_PI/4, max_dist)),
            Sector(Vector3f(0, 0, -1), Vector3f(1, 1, 0).normalized(), Vector3f(-M_PI/2, M_PI/4, min_dist), Vector3f(0, 3*M_PI/4, max_dist)),
            Sector(Vector3f(1, 1, 0).normalized(), Vector3f(0, 0, 1), Vector3f(-M_PI/2, M_PI/4, min_dist), Vector3f(0, 3*M_PI/4, max_dist)),
            // +X +Y sector
            Sector(Vector3f(0, 0, 1), Vector3f(1, -1, 0).normalized(), Vector3f(0, M_PI/4, min_dist), Vector3f(M_PI/2, 3*M_PI/4, max_dist)),
            Sector(Vector3f(1, -1, 0).normalized(), Vector3f(0, 0, -1), Vector3f(0, M_PI/4, min_dist), Vector3f(M_PI/2, 3*M_PI/4, max_dist)),
            Sector(Vector3f(0, 0, -1), Vector3f(-1, 1, 0).normalized(), Vector3f(0, M_PI/4, min_dist), Vector3f(M_PI/2, 3*M_PI/4, max_dist)),
            Sector(Vector3f(-1, 1, 0).normalized(), Vector3f(0, 0, 1), Vector3f(0, M_PI/4, min_dist), Vector3f(M_PI/2, 3*M_PI/4, max_dist)),
            // -X +Y sector
            Sector(Vector3f(0, 0, 1), Vector3f(-1, -1, 0).normalized(), Vector3f(M_PI/2, M_PI/4, min_dist), Vector3f(M_PI, 3*M_PI/4, max_dist)),
            Sector(Vector3f(-1, -1, 0).normalized(), Vector3f(0, 0, -1), Vector3f(M_PI/2, M_PI/4, min_dist), Vector3f(M_PI, 3*M_PI/4, max_dist)),
            Sector(Vector3f(0, 0, -1), Vector3f(1, 1, 0).normalized(), Vector3f(M_PI/2, M_PI/4, min_dist), Vector3f(M_PI, 3*M_PI/4, max_dist)),
            Sector(Vector3f(1, 1, 0).normalized(), Vector3f(0, 0, 1), Vector3f(M_PI/2, M_PI/4, min_dist), Vector3f(M_PI, 3*M_PI/4, max_dist))
        };

        auto nbLines = std::distance(lines_first, lines_last);

        // Sort lines into sectors
        std::array<LineIt, 32> line_sector_ends;
        {
            LineIt it_begin = lines_first;
            int idx = 0;
            for(Sector& sector : sectors)
            {
                for(LineIt it = it_begin; it < lines_last; ++it)
                {
                    const auto& line = (*it).*line_member;
                    if(sector.bounds.ContainsMoment(cart2spherical(line.m))
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
            for(Sector& sector : sectors)
            {
                LineIt it_end = line_sector_ends[idx];
                auto count = std::distance(it_begin, it_end);
                sector.rootNode = BuildNode(it_begin, it_end, sector.bounds, 0);

                it_begin = it_end;

                idx++;
            }
        }

        return Tree<Content, line_member>(nbLines, std::move(sectors));
    }
};

}
