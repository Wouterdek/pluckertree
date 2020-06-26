#include "DataSetGenerator.h"

std::vector<LineWrapper> GenerateRandomLines(std::random_device& dev, unsigned int seed, unsigned int lineCount, float maxDist)
{
    std::vector<LineWrapper> lines{};
    lines.reserve(lineCount);

    std::default_random_engine rng{seed};
    std::uniform_real_distribution<float> dist(0, maxDist);
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

std::vector<LineWrapper> GenerateParallelLines(std::random_device& dev, unsigned int seed, unsigned int lineCount, float maxDist, const Vector3f& direction)
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

std::vector<LineWrapper> GenerateEquiDistantLines(std::random_device& dev, unsigned int seed, unsigned int lineCount, float distance)
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

std::vector<LineWrapper> GenerateEqualMomentLines(std::random_device& dev, unsigned int seed, unsigned int lineCount, const Vector3f& moment)
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
        float angle = unitDist(dev) * 2.0f * M_PI;
        Vector3f d = std::sin(angle) * baseU + std::cos(angle) * baseV;
        lines.emplace_back(Line(d, moment));
    }
    return lines;
}

std::vector<LineSegmentWrapper> GenerateRandomLineSegments(std::random_device& dev, unsigned int seed, unsigned int lineCount, float maxDist, float minT, float maxT)
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
