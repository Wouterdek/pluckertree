#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <Pluckertree.h>
#include <iostream>

using namespace testing;

/*
 pluckertree::Line line(PI/2.0, Eigen::Vector3f(1.0, 0.0, 0.0));
std::array<pluckertree::Line, 2000> lines;

auto tree = pluckertree::TreeBuilder::Build(lines.begin(), lines.end());
tree.Add(line);

std::array<pluckertree::Line, 20> searchResult;
auto nbLinesFound = tree.FindNeighbours(Eigen::Vector3f(0, 0, 0), searchResult.begin(), searchResult.begin()+10);

*/

TEST(Tree, TestBuild)
{
}

TEST(Tree, TestAdd)
{
}

TEST(Tree, TestRemove)
{
}

TEST(Tree, TestFindNeighbours)
{
}
