#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <iostream>
#include <Pluckertree.h>
#include <random>
#include <fstream>
#include <string>
#include <thread>
#include <mutex>
#include "DataSetGenerator.h"

using namespace testing;
using namespace pluckertree;
using namespace Eigen;

const std::string output_path("/home/wouter/Documents/pluckertree_benchmarks/");

template<typename size_type>
void parallel_for(size_type arr_size, std::function<void(size_type)> f)
{
    auto threadCount = std::thread::hardware_concurrency();
    size_type batchSize, batchRemainder;
    size_type batchRemainderBegin;
    if(arr_size < threadCount)
    {
        batchSize = 1;
        batchRemainder = 0;
        batchRemainderBegin = arr_size;
    } else {
        batchSize = arr_size / threadCount;
        batchRemainder = arr_size % threadCount;
        batchRemainderBegin = arr_size - batchRemainder;
    }

    std::vector<std::thread> threads {};
    for(size_type i = 0; i < arr_size; i += batchSize)
    {
        threads.emplace_back([&f](size_type begin, size_type end){
            for(auto cur = begin; cur < end; ++cur)
            {
                f(cur);
            }
        }, i, i+batchSize);
    }

    for(auto cur = batchRemainderBegin; cur < arr_size; ++cur)
    {
        f(cur);
    }

    for(auto& thread : threads)
    {
        thread.join();
    }
}

TEST(Benchmarks, NearestNeighbour_Lines_Random)
{
    std::fstream out(output_path+"NearestNeighbour_Lines_Random.txt", std::fstream::out | std::fstream::app);
    if(out.fail())
    {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    std::vector<unsigned int> bench_line_counts = {100000, 250000, 500000, 750000, 1000000};

    for(int bench_type_i = 0; bench_type_i < bench_line_counts.size(); ++bench_type_i)
    {
        std::cout << "Testing bench type " << (bench_type_i+1) << " of " << bench_line_counts.size() << std::endl;

        out << "BENCH INFO" << std::endl;
        unsigned int line_count = bench_line_counts[bench_type_i];
        unsigned int query_count = 10;
        unsigned int iteration_count = 10;
        float max_dist = 100;

        out << "Line count: " << line_count << std::endl;
        out << "Query count: " << query_count << std::endl;
        out << "Iteration count: " << iteration_count << std::endl;
        out << "Maximum distance: " << max_dist << std::endl;

        out << "START BENCH" << std::endl;
        for(int i = 0; i < iteration_count; ++i)
        {
            std::cout << "Iteration " << (i+1) << " of " << iteration_count << std::endl;
            unsigned int query_seed, line_seed;
            {
                std::random_device dev{};
                query_seed = dev();
                line_seed = dev();
            }

            out << "Query generation seed: " << query_seed << std::endl;
            out << "Line generation seed: " << line_seed << std::endl;

            //Generate lines and tree
            std::cout << "Generating lines & tree\r";
            std::vector<LineWrapper> lines = GenerateRandomLines(line_seed, line_count, max_dist);
            auto tree = TreeBuilder<LineWrapper, &LineWrapper::l>::Build(lines.begin(), lines.end());

            //Generate query points
            std::cout << "Generating query points\r";
            auto query_points = GenerateRandomPoints(query_seed, query_count, max_dist);

            //Perform queries
            out << "START DATA" << std::endl;
            std::cout << "Running queries        \r";
            std::mutex mutex;
            parallel_for(query_points.size(), std::function([&query_points, &tree, &out, &mutex](std::vector<Vector3f>::size_type query_i){
                const auto& query = query_points[query_i];
                std::array<const LineWrapper *, 1> result{nullptr};
                float result_dist = 1E99;
                auto nbResultsFound = tree.FindNeighbours(query, result.begin(), result.end(), result_dist);

                {
                    const std::lock_guard<std::mutex> l(mutex);
                    out << Diag::visited << ";" << Diag::minimizations << std::endl;
                }
            }));
            std::cout << "                       \r";
            out << "END DATA" << std::endl;
            out.flush();
        }
        out << "END BENCH" << std::endl << std::endl;
    }

    out.flush();
    out.close();
}

TEST(Benchmarks, NearestHit_Lines_Random)
{
    std::fstream out(output_path+"NearestHit_Lines_Random.txt", std::fstream::out | std::fstream::app);
    if(out.fail())
    {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    std::vector<unsigned int> bench_line_counts = {100000, 250000, 500000, 750000, 1000000};

    for(int bench_type_i = 0; bench_type_i < bench_line_counts.size(); ++bench_type_i)
    {
        std::cout << "Testing bench type " << (bench_type_i+1) << " of " << bench_line_counts.size() << std::endl;

        out << "BENCH INFO" << std::endl;
        unsigned int line_count = bench_line_counts[bench_type_i];
        unsigned int query_count = 10;
        unsigned int iteration_count = 10;
        float max_dist = 100;

        out << "Line count: " << line_count << std::endl;
        out << "Query count: " << query_count << std::endl;
        out << "Iteration count: " << iteration_count << std::endl;
        out << "Maximum distance: " << max_dist << std::endl;

        out << "START BENCH" << std::endl;
        for(int i = 0; i < iteration_count; ++i)
        {
            std::cout << "Iteration " << (i+1) << " of " << iteration_count << std::endl;
            unsigned int query_seed, line_seed;
            {
                std::random_device dev{};
                query_seed = dev();
                line_seed = dev();
            }

            out << "Query generation seed: " << query_seed << std::endl;
            out << "Line generation seed: " << line_seed << std::endl;

            //Generate lines and tree
            std::cout << "Generating lines & tree\r";
            std::vector<LineWrapper> lines = GenerateRandomLines(line_seed, line_count, max_dist);
            auto tree = TreeBuilder<LineWrapper, &LineWrapper::l>::Build(lines.begin(), lines.end());

            //Generate query points
            std::cout << "Generating query points\r";
            auto query_points = GenerateRandomPoints(query_seed, query_count, max_dist);
            auto query_normals = GenerateRandomNormals(query_seed, line_count);

            //Perform queries
            out << "START DATA" << std::endl;
            std::cout << "Running queries        \r";
            std::mutex mutex;
            parallel_for(query_points.size(), std::function([&query_points, &query_normals, &tree, &out, &mutex](std::vector<Vector3f>::size_type query_i){
                const auto& query = query_points[query_i];
                const auto& query_normal = query_points[query_i];
                std::array<const LineWrapper *, 1> result{nullptr};
                float result_dist = 1E99;
                auto nbResultsFound = tree.FindNearestHits(query, query_normal, result.begin(), result.end(), result_dist);

                {
                    const std::lock_guard<std::mutex> l(mutex);
                    out << Diag::visited << ";" << Diag::minimizations << std::endl;
                }
            }));
            std::cout << "                       \r";
            out << "END DATA" << std::endl;
            out.flush();
        }
        out << "END BENCH" << std::endl << std::endl;
    }

    out.flush();
    out.close();
}