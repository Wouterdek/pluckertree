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

std::vector<LineWrapper> GenerateRandomLines(unsigned int seed, unsigned int lineCount, float maxDist);
std::vector<LineWrapper> GenerateParallelLines(unsigned int seed, unsigned int lineCount, float maxDist, const Vector3f& direction);
std::vector<LineWrapper> GenerateEquiDistantLines(unsigned int seed, unsigned int lineCount, float maxDist); //Equidistant from center
std::vector<LineWrapper> GenerateEqualMomentLines(unsigned int seed, unsigned int lineCount, const Vector3f& moment);
std::vector<LineSegmentWrapper> GenerateRandomLineSegments(unsigned int seed, unsigned int lineCount, float maxDist, float minT, float maxT);

std::vector<Vector3f> GenerateRandomPoints(unsigned int seed, unsigned int pointCount, float maxDist);
std::vector<Vector3f> GenerateRandomNormals(unsigned int seed, unsigned int normalCount);