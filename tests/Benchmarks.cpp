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
        threadCount = arr_size;
    } else {
        batchSize = arr_size / threadCount;
        batchRemainder = arr_size % threadCount;
        batchRemainderBegin = arr_size - batchRemainder;
    }

    std::vector<std::thread> threads {};
    for(size_type i = 0; i < threadCount; ++i)
    {
        threads.emplace_back([&f](size_type begin, size_type end){
            for(auto cur = begin; cur < end; ++cur)
            {
                f(cur);
            }
        }, i*batchSize, (i+1)*batchSize);
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

TEST(Benchmarks, DISABLED_NearestNeighbour_Lines_Random)
{
    std::fstream out(output_path+"NearestNeighbour_Lines_Random1.txt", std::fstream::out | std::fstream::app);
    if(out.fail())
    {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    std::vector<unsigned int> bench_line_counts = {100, 250000, 500000, 750000, 1000000};

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

TEST(Benchmarks, DISABLED_NearestHit_Lines_Random)
{
    std::fstream out(output_path+"NearestHit_Lines_Random.txt", std::fstream::out | std::fstream::app);
    if(out.fail())
    {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    std::vector<unsigned int> bench_line_counts = {100, 250000, 500000, 750000, 1000000};

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

TEST(Benchmarks, DISABLED_NearestNeighbour_LineSegments_Random)
{
    std::fstream out(output_path+"NearestNeighbour_LineSegments_Random.txt", std::fstream::out | std::fstream::app);
    if(out.fail())
    {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    std::vector<unsigned int> bench_line_counts = {100, 250000, 500000, 750000, 1000000};

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
            auto lines = GenerateRandomLineSegments(line_seed, line_count, max_dist, -100, 100);
            auto tree = segments::TreeBuilder<LineSegmentWrapper, &LineSegmentWrapper::l>::Build(lines.begin(), lines.end());

            //Generate query points
            std::cout << "Generating query points\r";
            auto query_points = GenerateRandomPoints(query_seed, query_count, max_dist);

            //Perform queries
            out << "START DATA" << std::endl;
            std::cout << "Running queries        \r";
            std::mutex mutex;
            parallel_for(query_points.size(), std::function([&query_points, &tree, &out, &mutex](std::vector<Vector3f>::size_type query_i){
                const auto& query = query_points[query_i];
                std::array<const LineSegmentWrapper *, 1> result{nullptr};
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

TEST(Benchmarks, DISABLED_NearestHit_LineSegments_Random)
{
    std::fstream out(output_path+"NearestHit_LineSegments_Random.txt", std::fstream::out | std::fstream::app);
    if(out.fail())
    {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    std::vector<unsigned int> bench_line_counts = {100, 250000, 500000, 750000, 1000000};

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
            auto lines = GenerateRandomLineSegments(line_seed, line_count, max_dist, -100, 100);
            auto tree = segments::TreeBuilder<LineSegmentWrapper, &LineSegmentWrapper::l>::Build(lines.begin(), lines.end());

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
                std::array<const LineSegmentWrapper *, 1> result{nullptr};
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


TEST(Benchmarks, DISABLED_NearestNeighbour_Lines_Parallel)
{
    std::fstream out(output_path+"NearestNeighbour_Lines_Parallel.txt", std::fstream::out | std::fstream::app);
    if(out.fail())
    {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    std::vector<unsigned int> bench_line_counts = {100, 250000, 500000, 750000, 1000000};

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
            std::default_random_engine rng{line_seed+1};
            std::uniform_real_distribution<float> dist(0, 1);
            Vector3f direction(dist(rng), dist(rng), dist(rng));
            direction.normalize();
            std::vector<LineWrapper> lines = GenerateParallelLines(line_seed, line_count, max_dist, direction);
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

TEST(Benchmarks, DISABLED_NearestNeighbour_Lines_EquiDistant)
{
    std::fstream out(output_path+"NearestNeighbour_Lines_EquiDistant.txt", std::fstream::out | std::fstream::app);
    if(out.fail())
    {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    std::vector<unsigned int> bench_line_counts = {100, 250000, 500000, 750000, 1000000};

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
            std::vector<LineWrapper> lines = GenerateEquiDistantLines(line_seed, line_count, max_dist);
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

TEST(Benchmarks, DISABLED_NearestNeighbour_Lines_EqualMoment)
{
    std::fstream out(output_path+"NearestNeighbour_Lines_EqualMoment.txt", std::fstream::out | std::fstream::app);
    if(out.fail())
    {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    std::vector<unsigned int> bench_line_counts = {100, 250000, 500000, 750000, 1000000};

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
            std::default_random_engine rng{line_seed+1};
            std::uniform_real_distribution<float> dist(0, 1);
            Vector3f moment(dist(rng), dist(rng), dist(rng));
            moment.normalize();
            moment *= dist(rng) * max_dist;
            std::vector<LineWrapper> lines = GenerateEqualMomentLines(line_seed, line_count, moment);
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

TEST(Benchmarks, DISABLED_NearestNeighbour_LineSegments_VaryingLengths)
{
    std::fstream out(output_path+"NearestNeighbour_LineSegments_VaryingLengths.txt", std::fstream::out | std::fstream::app);
    if(out.fail())
    {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    unsigned int line_count = 100000;
    unsigned int query_seed, line_seed;
    {
        std::random_device dev{};
        query_seed = dev();
        line_seed = dev();
    }

    // Generate lines
    float max_dist = 100;
    auto lines = GenerateRandomLineSegments(line_seed, line_count, max_dist, -100, 100);

    //Generate query points
    unsigned int query_count = 100;
    auto query_points = GenerateRandomPoints(query_seed, query_count, max_dist);
    //auto query_normals = GenerateRandomNormals(query_seed, line_count);

    std::vector<float> bench_lineseg_t_min = {-100, -75, -50, -25, -1};
    std::vector<float> bench_lineseg_t_max = { 100,  75,  50,  25,  1};

    for(int bench_type_i = 0; bench_type_i < bench_lineseg_t_min.size(); ++bench_type_i)
    {
        std::cout << "Testing bench type " << (bench_type_i+1) << " of " << bench_lineseg_t_min.size() << std::endl;

        out << "BENCH INFO" << std::endl;
        float t_min = bench_lineseg_t_min[bench_type_i];
        float t_max = bench_lineseg_t_max[bench_type_i];

        out << "Query generation seed: " << query_seed << std::endl;
        out << "Line generation seed: " << line_seed << std::endl;
        out << "Line count: " << line_count << std::endl;
        out << "Query count: " << query_count << std::endl;
        out << "Maximum distance: " << max_dist << std::endl;
        out << "T min: " << t_min << std::endl;
        out << "T max: " << t_max << std::endl;

        out << "START BENCH" << std::endl;

        //Modifying segment lengths
        for(auto& s : lines)
        {
            s.l.t1 = t_min;
            s.l.t2 = t_max;
        }

        //Generate lines and tree
        std::cout << "Generating tree\r";
        auto tree = segments::TreeBuilder<LineSegmentWrapper, &LineSegmentWrapper::l>::Build(lines.begin(), lines.end());

        //Perform queries
        out << "START DATA" << std::endl;
        std::cout << "Running queries        \r";
        std::mutex mutex;
        parallel_for(query_points.size(), std::function([&query_points, &tree, &out, &mutex](std::vector<Vector3f>::size_type query_i){
            const auto& query = query_points[query_i];
            std::array<const LineSegmentWrapper *, 1> result{nullptr};
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

        out << "END BENCH" << std::endl << std::endl;
    }

    out.flush();
    out.close();
}

TEST(Benchmarks, DISABLED_NearestNeighbour_Lines_RequestSize)
{
    std::fstream out(output_path+"NearestNeighbour_Lines_RequestSize.txt", std::fstream::out | std::fstream::app);
    if(out.fail())
    {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    unsigned int line_count = 100000;
    unsigned int query_count = 100;
    unsigned int query_seed, line_seed;
    {
        std::random_device dev{};
        query_seed = dev();
        line_seed = dev();
    }

    // Generate lines
    float max_dist = 100;
    auto lines = GenerateRandomLines(line_seed, line_count, max_dist);
    auto tree = TreeBuilder<LineWrapper, &LineWrapper::l>::Build(lines.begin(), lines.end());

    //Generate query points
    auto query_points = GenerateRandomPoints(query_seed, query_count, max_dist);

    std::vector<unsigned int> bench_request_sizes = {1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50};

    for(int bench_type_i = 0; bench_type_i < bench_request_sizes.size(); ++bench_type_i)
    {
        std::cout << "Testing bench type " << (bench_type_i+1) << " of " << bench_request_sizes.size() << std::endl;

        out << "BENCH INFO" << std::endl;
        unsigned int request_size = bench_request_sizes[bench_type_i];

        float max_dist = 100;

        out << "Line count: " << line_count << std::endl;
        out << "Query count: " << query_count << std::endl;
        out << "Maximum distance: " << max_dist << std::endl;
        out << "Request size: " << request_size << std::endl;

        out << "START BENCH" << std::endl;

        out << "Query generation seed: " << query_seed << std::endl;
        out << "Line generation seed: " << line_seed << std::endl;

        //Perform queries
        out << "START DATA" << std::endl;
        std::cout << "Running queries        \r";
        std::mutex mutex;
        parallel_for(query_points.size(), std::function([request_size, &query_points, &tree, &out, &mutex](std::vector<Vector3f>::size_type query_i){
            const auto& query = query_points[query_i];
            std::vector<const LineWrapper *> result(request_size);
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

        out << "END BENCH" << std::endl << std::endl;
    }

    out.flush();
    out.close();
}

TEST(Benchmarks, DISABLED_NearestNeighbour_Lines_Origin)
{
    std::fstream out(output_path+"NearestNeighbour_Lines_Origin.txt", std::fstream::out | std::fstream::app);
    if(out.fail())
    {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    unsigned int line_count = 100000;
    unsigned int query_count = 100;
    unsigned int query_seed, line_seed;
    {
        std::random_device dev{};
        query_seed = dev();
        line_seed = dev();
    }

    // Generate lines
    float max_line_dist = 20;
    auto lines = GenerateRandomLines(line_seed, line_count, max_line_dist);
    std::vector<LineWrapper> translated_lines = lines;

    //Generate query points
    float max_query_dist = 100;
    auto query_points = GenerateRandomPoints(query_seed, query_count, max_query_dist);

    //Generate translation vectors
    std::vector<Vector3f> bench_translation_vectors {
        Vector3f(0, 0, 0),
        Vector3f(20, 0, 0),
        Vector3f(40, 0, 0),
        Vector3f(60, 0, 0),
        Vector3f(0, 0, 20),
        Vector3f(0, 0, 40),
        Vector3f(0, 0, 60),
        Vector3f(std::sqrt(20), std::sqrt(20), 0),
    };

    for(int bench_type_i = 0; bench_type_i < bench_translation_vectors.size(); ++bench_type_i)
    {
        std::cout << "Testing bench type " << (bench_type_i+1) << " of " << bench_translation_vectors.size() << std::endl;

        out << "BENCH INFO" << std::endl;
        auto translation_vect = bench_translation_vectors[bench_type_i];
        out << "Line count: " << line_count << std::endl;
        out << "Query count: " << query_count << std::endl;
        out << "Maximum line distance: " << max_line_dist << std::endl;
        out << "Maximum query distance: " << max_query_dist << std::endl;
        out << "Translation vector: " << translation_vect.transpose() << std::endl;

        for(unsigned int i = 0; i < line_count; ++i)
        {
            Vector3f curP = lines[i].l.d.cross(lines[i].l.m);
            Vector3f p1 = curP + translation_vect;
            Vector3f p2 = (curP + lines[i].l.d) + translation_vect;
            translated_lines[i] = LineWrapper(Line::FromTwoPoints(p1, p2));
        }
        auto tree = TreeBuilder<LineWrapper, &LineWrapper::l>::Build(translated_lines.begin(), translated_lines.end());

        out << "START TREEDEPTH DATA" << std::endl;
        std::function<void(const TreeNode<LineWrapper, &LineWrapper::l>*, unsigned int)> visitor;
        visitor = [&out, &visitor](const TreeNode<LineWrapper, &LineWrapper::l>* curNode, unsigned int curDepth){
            if(curNode == nullptr)
            {
                return;
            }
            out << curDepth << std::endl;
            visitor(curNode->children[0].get(), curDepth+1);
            visitor(curNode->children[1].get(), curDepth+1);
        };
        for(const auto& sector : tree.sectors)
        {
            visitor(sector.rootNode.get(), 1);
        }
        out << "END TREEDEPTH DATA" << std::endl;

        out << "START BENCH" << std::endl;

        out << "Query generation seed: " << query_seed << std::endl;
        out << "Line generation seed: " << line_seed << std::endl;


        //Perform queries
        out << "START DATA" << std::endl;
        std::cout << "Running queries        \r";
        std::mutex mutex;
        parallel_for(query_points.size(), std::function([&query_points, &tree, &out, &mutex](std::vector<Vector3f>::size_type query_i){
            const auto& query = query_points[query_i];
            std::array<const LineWrapper *, 1> result {nullptr};
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

        out << "END BENCH" << std::endl << std::endl;
    }

    out.flush();
    out.close();
}


TEST(Benchmarks, DISABLED_NearestNeighbour_Lines_Dist_Histogram)
{
    std::fstream out(output_path+"NearestNeighbour_Lines_Dist_Histogram.txt", std::fstream::out | std::fstream::app);
    if(out.fail())
    {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    unsigned int line_count = 100000;
    unsigned int query_count = 100;
    float max_dist = 100;

    unsigned int query_seed, line_seed;
    {
        std::random_device dev{};
        query_seed = dev();
        line_seed = dev();
    }

    auto lines = GenerateRandomLines(line_seed, line_count, max_dist);
    auto query_points = GenerateRandomPoints(query_seed, query_count, max_dist);
    std::vector<float> dists(lines.size());

    for(const auto& q : query_points)
    {
        out << "BENCH INFO" << std::endl;
        out << "Line count: " << line_count << std::endl;
        out << "Query count: " << query_count << std::endl;
        out << "Maximum distance: " << max_dist << std::endl;
        out << "Query generation seed: " << query_seed << std::endl;
        out << "Line generation seed: " << line_seed << std::endl;

        out << "START BENCH" << std::endl;

        parallel_for(lines.size(), std::function([&lines, q, &dists](std::vector<Vector3f>::size_type line_i){
            const auto& l = lines[line_i].l;
            dists[line_i] = (q.cross(l.d) - l.m).norm();
        }));
        for(float dist : dists)
        {
            out << dist << std::endl;
        }

        out << "END DATA" << std::endl;

        out << "END BENCH" << std::endl << std::endl;
    }

    out.flush();
    out.close();
}

TEST(Benchmarks, DISABLED_NearestNeighbour_Lines_Dist_Progress)
{
    std::fstream out(output_path+"NearestNeighbour_Lines_Dist_Progress.txt", std::fstream::out | std::fstream::app);
    if(out.fail())
    {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    unsigned int line_count = 100000;
    unsigned int query_count = 100;
    float max_dist = 100;

    unsigned int query_seed, line_seed;
    {
        std::random_device dev{};
        query_seed = dev();
        line_seed = dev();
    }

    auto lines = GenerateRandomLines(line_seed, line_count, max_dist);
    auto tree = TreeBuilder<LineWrapper, &LineWrapper::l>::Build(lines.begin(), lines.end());
    auto query_points = GenerateRandomPoints(query_seed, query_count, max_dist);

    unsigned int query_i = 0;
    unsigned int time_i = 0;
    Diag::on_node_visited = std::function([&out, &query_i, &time_i](float best_so_far_dist, float cur_line_dist, float cur_node_mindist, int cur_depth)
    {
        out << query_i << ";" << time_i << ";" << best_so_far_dist << ';' << cur_line_dist << ";" << cur_node_mindist << ';' << cur_depth << std::endl;
        time_i++;
    });

    out << "BENCH INFO" << std::endl;
    out << "Line count: " << line_count << std::endl;
    out << "Query count: " << query_count << std::endl;
    out << "Maximum distance: " << max_dist << std::endl;
    out << "Query generation seed: " << query_seed << std::endl;
    out << "Line generation seed: " << line_seed << std::endl;
    out << "START BENCH" << std::endl;
    //for(const auto& q : query_points)
    for(query_i = 0; query_i < query_points.size(); query_i++)
    {
        const auto& q = query_points[query_i];
        std::cout << query_i << "/" << query_points.size() << std::endl;
        std::array<const LineWrapper *, 1> result{nullptr};
        float result_dist = 1E99;
        tree.FindNeighbours(q, result.begin(), result.end(), result_dist);
        time_i = 0;
    }
    out << "END BENCH" << std::endl << std::endl;

    out.flush();
    out.close();

    Diag::reset();
}

TEST(Benchmarks, DISABLED_NearestNeighbour_Lines_Dist_Tightness)
{
    std::fstream out(output_path+"NearestNeighbour_Lines_Dist_Tightness.txt", std::fstream::out | std::fstream::app);
    if(out.fail())
    {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    unsigned int line_count = 10000;
    unsigned int iterations = 10;
    float max_dist = 100;

    Diag::force_visit_all = true;

    unsigned int iter_i = 0;
    std::vector<float> min_bin_dist {};
    Diag::on_node_enter = std::function([&min_bin_dist](float best_so_far_dist, float cur_node_mindist, int cur_depth)
        {
            min_bin_dist.push_back(cur_node_mindist);
        });
    Diag::on_node_leave = std::function([&min_bin_dist](float best_so_far_dist, float cur_line_dist, float cur_node_mindist, int cur_depth)
        {
            min_bin_dist.pop_back();
        });
    Diag::on_node_visited = std::function([&out, &iter_i, &min_bin_dist](float best_so_far_dist, float cur_line_dist, float cur_node_mindist, int cur_depth)
      {
          for(int i = 0; i < min_bin_dist.size(); i++)
          {
              out << iter_i << ';' << cur_line_dist-min_bin_dist[i] << ';' << i << std::endl;
          }
      });

    out << "BENCH INFO" << std::endl;
    out << "Line count: " << line_count << std::endl;
    out << "Maximum distance: " << max_dist << std::endl;
    out << "START BENCH" << std::endl;
    std::random_device dev{};
    for(iter_i = 0; iter_i < iterations; ++iter_i)
    {
        unsigned int query_seed, line_seed;
        {
            query_seed = dev();
            line_seed = dev();
        }

        out << "Query generation seed: " << query_seed << std::endl;
        out << "Line generation seed: " << line_seed << std::endl;

        auto lines = GenerateRandomLines(line_seed, line_count, max_dist);
        auto tree = TreeBuilder<LineWrapper, &LineWrapper::l>::Build(lines.begin(), lines.end());
        auto query_point = GenerateRandomPoints(query_seed, 1, max_dist)[0];

        out << "START DATA" << std::endl;
        std::array<const LineWrapper *, 1> result{nullptr};
        float result_dist = 1E99;
        tree.FindNeighbours(query_point, result.begin(), result.end(), result_dist);
        out << "END DATA" << std::endl;
    }
    out << "END BENCH" << std::endl << std::endl;

    out.flush();
    out.close();

    Diag::reset();
}

TEST(Benchmarks, NearestNeighbour_Lines_Random_SplitTypes)
{
    std::fstream out(output_path+"NearestNeighbour_Lines_Random_SplitTypes.txt", std::fstream::out | std::fstream::app);
    if(out.fail())
    {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    unsigned int line_count = 100000;

    {
        out << "BENCH INFO" << std::endl;
        unsigned int iteration_count = 100;
        float max_dist = 100;

        out << "Line count: " << line_count << std::endl;
        out << "Iteration count: " << iteration_count << std::endl;
        out << "Maximum distance: " << max_dist << std::endl;

        out << "START BENCH" << std::endl;
        for(int i = 0; i < iteration_count; ++i)
        {
            std::cout << "Iteration " << (i+1) << " of " << iteration_count << std::endl;
            unsigned int line_seed;
            {
                std::random_device dev{};
                line_seed = dev();
            }

            out << "Line generation seed: " << line_seed << std::endl;

            //Generate lines and tree
            std::cout << "Generating lines & tree\r";
            std::vector<LineWrapper> lines = GenerateRandomLines(line_seed, line_count, max_dist);
            auto tree = TreeBuilder<LineWrapper, &LineWrapper::l>::Build(lines.begin(), lines.end());


            //Perform queries
            out << "START DATA" << std::endl;
            std::cout << "Running queries        \r";
            std::array<unsigned int, 3> moment_splits {0, 0, 0};
            unsigned int directional_splits = 0;
            std::function<void(const TreeNode<LineWrapper, &LineWrapper::l>*)> visitor;
            visitor = std::function([&visitor, &moment_splits, &directional_splits](const TreeNode<LineWrapper, &LineWrapper::l>* node){
                if(node == nullptr)
                {
                    return;
                }

                if(node->type == NodeType::moment)
                {
                    moment_splits[node->bound_component_idx]++;
                }else if(node->type == NodeType::direction)
                {
                    directional_splits++;
                }

                visitor(node->children[0].get());
                visitor(node->children[1].get());
            });
            for(const auto& sector : tree.sectors)
            {
                visitor(sector.rootNode.get());
            }
            unsigned int total = moment_splits[0] + moment_splits[1] + moment_splits[2] + directional_splits;
            out << (float)moment_splits[0]/(float)total << ";" << (float)moment_splits[1]/(float)total << ";" << (float)moment_splits[2]/(float)total << ";" << (float)directional_splits/(float)total << std::endl;
            std::cout << "                       \r";
            out << "END DATA" << std::endl;
            out.flush();
        }
        out << "END BENCH" << std::endl << std::endl;
    }

    out.flush();
    out.close();
}
