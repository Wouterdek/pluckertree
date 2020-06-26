#pragma once

#include <Pluckertree.h>
#include <PluckertreeSegments.h>
#include <Eigen/Dense>
#include <random>

using Line = pluckertree::Line;
using LineSegment = pluckertree::segments::LineSegment;
using Eigen::Vector3f;

struct LineWrapper
{
    Line l;
    explicit LineWrapper(Line line) : l(std::move(line)) {}
};

struct LineSegmentWrapper
{
    LineSegment l;
    explicit LineSegmentWrapper(LineSegment line) : l(std::move(line)) {}
};

std::vector<LineWrapper> GenerateRandomLines(std::random_device& dev, unsigned int seed, unsigned int lineCount, float maxDist);
std::vector<LineWrapper> GenerateParallelLines(std::random_device& dev, unsigned int seed, unsigned int lineCount, float maxDist);
std::vector<LineWrapper> GenerateEquiDistantLines(std::random_device& dev, unsigned int seed, unsigned int lineCount, float maxDist); //Equidistant from center
std::vector<LineWrapper> GenerateEqualMomentLines(std::random_device& dev, unsigned int seed, unsigned int lineCount, float maxDist);
std::vector<LineSegmentWrapper> GenerateRandomLineSegments(std::random_device& dev, unsigned int seed, unsigned int lineCount, float maxDist, float minT, float maxT);