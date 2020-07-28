#include "DataSetGenerator.h"
#include <fstream>
#include <Eigen/Dense>

std::vector<LineWrapper> LoadFromFile(const std::string& filename)
{
    std::vector<LineWrapper> result;
    std::ifstream in(filename);
    std::string line;
    while (std::getline(in, line))
    {
        std::istringstream iss(line);
        char sep;
        Eigen::Vector3f m, d;
        if (!(iss >> m.x() >> sep >> m.y()>> sep >> m.z() >> sep >> d.x() >> sep >> d.y() >> sep >> d.z())) { break; } // error

        result.emplace_back(Line(d, m));
    }
    return result;
}

std::vector<LineWrapper> GenerateRandomLines(unsigned int seed, unsigned int lineCount, float maxDist)
{
    std::vector<LineWrapper> lines{};
    lines.reserve(lineCount);

    std::default_random_engine rng{seed};
    std::uniform_real_distribution<float> dist(-maxDist, maxDist);
    for (int i = 0; i < lineCount; ++i) {
        Line l(Vector3f::Zero(), Vector3f::Zero());
        do {
            l = Line::FromTwoPoints(
                    Vector3f(dist(rng), dist(rng), dist(rng)),
                    Vector3f(dist(rng), dist(rng), dist(rng))
            );
        }while(l.m.norm() >= 150);
        lines.emplace_back(l);
    }
    return lines;
}

std::vector<LineWrapper> GenerateParallelLines(unsigned int seed, unsigned int lineCount, float maxDist, const Vector3f& direction)
{
    std::vector<LineWrapper> lines{};
    lines.reserve(lineCount);

    std::default_random_engine rng{seed};
    std::uniform_real_distribution<float> dist(0, maxDist);
    for (int i = 0; i < lineCount; ++i) {
        Line l(Vector3f::Zero(), Vector3f::Zero());
        do {
            l = Line::FromPointAndDirection(
                    Vector3f(dist(rng), dist(rng), dist(rng)),
                    direction
            );
        }while(l.m.norm() >= 150);
        lines.emplace_back(l);
    }
    return lines;
}

std::vector<LineWrapper> GenerateEquiDistantLines(unsigned int seed, unsigned int lineCount, float distance)
{
    std::vector<LineWrapper> lines{};
    lines.reserve(lineCount);

    std::default_random_engine rng{seed};
    std::uniform_real_distribution<float> dist(0, 1.0);
    for (int i = 0; i < lineCount; ++i) {
        Vector3f p = Vector3f(dist(rng), dist(rng), dist(rng));
        p.normalize();
        p *= distance;
        Vector3f d = Vector3f(dist(rng), dist(rng), dist(rng));
        d.normalize();
        lines.emplace_back(Line::FromPointAndDirection(p, d));
    }
    return lines;
}

std::vector<LineWrapper> GenerateEqualMomentLines(unsigned int seed, unsigned int lineCount, const Vector3f& moment)
{
    std::vector<LineWrapper> lines{};
    lines.reserve(lineCount);

    std::default_random_engine rng{seed};
    std::uniform_real_distribution<float> unitDist(0, 1.0f);

    Vector3f moment_d = moment.normalized();
    Vector3f baseU = Vector3f(unitDist(rng), unitDist(rng), unitDist(rng));
    baseU -= moment_d * moment.dot(baseU);
    baseU.normalize();
    Vector3f baseV = baseU.cross(moment_d);

    for (int i = 0; i < lineCount; ++i) {
        float angle = unitDist(rng) * 2.0f * M_PI;
        Vector3f d = std::sin(angle) * baseU + std::cos(angle) * baseV;
        lines.emplace_back(Line(d, moment));
    }
    return lines;
}

std::vector<LineSegmentWrapper> GenerateRandomLineSegments(unsigned int seed, unsigned int lineCount, float maxDist, float minT, float maxT)
{
    std::vector<LineSegmentWrapper> lines{};
    lines.reserve(lineCount);

    std::default_random_engine rng{seed};
    std::uniform_real_distribution<float> dist(0, maxDist);
    std::uniform_real_distribution<float> distT(minT, maxT);
    for (int i = 0; i < lineCount; ++i) {
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

    return lines;
}

std::vector<Vector3f> GenerateRandomPoints(unsigned int seed, unsigned int pointCount, float maxDist)
{
    std::vector<Vector3f> queryPoints{};
    queryPoints.reserve(pointCount);
    std::default_random_engine rng{seed};
    std::uniform_real_distribution<float> dist(0, maxDist);

    for (int query_i = 0; query_i < pointCount; ++query_i) {
        queryPoints.emplace_back(dist(rng), dist(rng), dist(rng));
    }
    return queryPoints;
}

std::vector<Vector3f> GenerateRandomNormals(unsigned int seed, unsigned int normalCount)
{
    std::vector<Vector3f> query_point_normals {};
    query_point_normals.reserve(normalCount);

    std::default_random_engine rng {seed+1};
    std::uniform_real_distribution<float> dist(0, 1);

    for(int i = 0; i < normalCount; ++i)
    {
        query_point_normals.emplace_back(dist(rng), dist(rng), dist(rng));
        query_point_normals.back().normalize();
    }

    return query_point_normals;
}