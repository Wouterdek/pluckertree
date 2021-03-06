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

//auto lines = LoadFromFile("/home/wouter/Documents/sphere_lines.txt");


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
    std::random_device dev{};

    bool isLogPlot = true;
    bool isSqrtPlot = false;

    std::string filename;
    std::vector<unsigned int> bench_line_counts;
    if(isLogPlot)
    {
        filename = output_path+"NearestNeighbour_Lines_Random_log.txt";
        bench_line_counts = {100, 1000, 10000, 100000, 1000000};
    }
    else if(isSqrtPlot)
    {
        filename = output_path+"NearestNeighbour_Lines_Random_sqrt.txt";
        bench_line_counts = {40000, 90000, 160000, 250000, 360000};
    }
    else
    {
        filename = output_path+"NearestNeighbour_Lines_Random.txt";
        bench_line_counts = {100, 250000, 500000, 750000, 1000000};
    }

    std::fstream out(filename, std::fstream::out | std::fstream::app);
    if(out.fail())
    {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    for(int bench_type_i = 0; bench_type_i < bench_line_counts.size(); ++bench_type_i)
    {
        std::cout << "Testing bench type " << (bench_type_i+1) << " of " << bench_line_counts.size() << std::endl;

        out << "BENCH INFO" << std::endl;
        unsigned int line_count = bench_line_counts[bench_type_i];
        unsigned int query_count = 100;
        float max_dist = 100;

        out << "Line count: " << line_count << std::endl;
        out << "Query count: " << query_count << std::endl;
        out << "Maximum distance: " << max_dist << std::endl;

        out << "START BENCH" << std::endl;
        std::mutex mutex;
        parallel_for(query_count, std::function([&out, &mutex, &dev, max_dist, query_count, line_count](unsigned int query_i){
            unsigned int query_seed, line_seed;
            {
                const std::lock_guard<std::mutex> l(mutex);
                query_seed = dev();
                line_seed = dev();
            }

            //Generate line, tree and query
            std::vector<LineWrapper> lines = GenerateRandomLines(line_seed, line_count, max_dist);
            auto tree = TreeBuilder<LineWrapper, &LineWrapper::l>::Build(lines.begin(), lines.end());
            auto query_point = GenerateRandomPoints(query_seed, 1, max_dist)[0];

            //Perform query
            std::array<const LineWrapper *, 1> result{nullptr};
            float result_dist = 1E99;
            auto nbResultsFound = tree.FindNeighbours(query_point, result.begin(), result.end(), result_dist);

            {
                const std::lock_guard<std::mutex> l(mutex);
                out << Diag::visited << ";" << Diag::minimizations << std::endl;
            }
        }));
        out.flush();
        out << "END BENCH" << std::endl << std::endl;
    }

    out.flush();
    out.close();
}

TEST(Benchmarks, DISABLED_NearestHit_Lines_Random)
{
    std::random_device dev{};
    std::fstream out(output_path+"NearestHit_Lines_Random.txt", std::fstream::out | std::fstream::app);
    if(out.fail())
    {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    //std::vector<unsigned int> bench_line_counts = {100, 250000, 500000, 750000, 1000000};
    std::vector<unsigned int> bench_line_counts = {100, 1000, 10000, 100000, 1000000};

    for(int bench_type_i = 0; bench_type_i < bench_line_counts.size(); ++bench_type_i)
    {
        std::cout << "Testing bench type " << (bench_type_i+1) << " of " << bench_line_counts.size() << std::endl;

        out << "BENCH INFO" << std::endl;
        unsigned int line_count = bench_line_counts[bench_type_i];
        unsigned int query_count = 100;
        float max_dist = 100;

        out << "Line count: " << line_count << std::endl;
        out << "Query count: " << query_count << std::endl;
        out << "Maximum distance: " << max_dist << std::endl;

        out << "START BENCH" << std::endl;
        std::mutex mutex;
        parallel_for(query_count, std::function([&out, &mutex, &dev, max_dist, query_count, line_count](unsigned int query_i){
            unsigned int query_seed, line_seed;
            {
                const std::lock_guard<std::mutex> l(mutex);
                query_seed = dev();
                line_seed = dev();
            }

            //Generate line, tree, query
            std::vector<LineWrapper> lines = GenerateRandomLines(line_seed, line_count, max_dist);
            auto tree = TreeBuilder<LineWrapper, &LineWrapper::l>::Build(lines.begin(), lines.end());
            auto query_point = GenerateRandomPoints(query_seed, 1, max_dist)[0];
            auto query_normal = GenerateRandomNormals(query_seed+1, 1)[0];

            //Perform query
            std::array<const LineWrapper *, 1> result{nullptr};
            float result_dist = 1E99;
            auto nbResultsFound = tree.FindNearestHits(query_point, query_normal, result.begin(), result.end(), result_dist);

            {
                const std::lock_guard<std::mutex> l(mutex);
                out << Diag::visited << ";" << Diag::minimizations << std::endl;
            }
        }));
        out.flush();

        out << "END BENCH" << std::endl << std::endl;
    }

    out.flush();
    out.close();
}

TEST(Benchmarks, DISABLED_NearestNeighbour_LineSegments_Random)
{
    std::random_device dev{};

    std::fstream out(output_path+"NearestNeighbour_LineSegments_Random.txt", std::fstream::out | std::fstream::app);
    if(out.fail())
    {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    //std::vector<unsigned int> bench_line_counts = {100, 250000, 500000, 750000, 1000000};
    std::vector<unsigned int> bench_line_counts = {100, 1000, 10000, 100000, 1000000};

    for(int bench_type_i = 0; bench_type_i < bench_line_counts.size(); ++bench_type_i)
    {
        std::cout << "Testing bench type " << (bench_type_i+1) << " of " << bench_line_counts.size() << std::endl;

        out << "BENCH INFO" << std::endl;
        unsigned int line_count = bench_line_counts[bench_type_i];
        unsigned int query_count = 100;
        float max_dist = 100;

        out << "Line count: " << line_count << std::endl;
        out << "Query count: " << query_count << std::endl;
        out << "Maximum distance: " << max_dist << std::endl;

        out << "START BENCH" << std::endl;
        std::mutex mutex;
        parallel_for(query_count, std::function([&out, &mutex, &dev, max_dist, query_count, line_count](unsigned int query_i){
            unsigned int query_seed, line_seed;
            {
                const std::lock_guard<std::mutex> l(mutex);
                query_seed = dev();
                line_seed = dev();
            }

            //Generate line, tree, query
            auto lines = GenerateRandomLineSegments(line_seed, line_count, max_dist, -100, 100);
            auto tree = segments::TreeBuilder<LineSegmentWrapper, &LineSegmentWrapper::l>::Build(lines.begin(), lines.end());
            auto query_point = GenerateRandomPoints(query_seed, 1, max_dist)[0];

            //Perform query
            std::array<const LineSegmentWrapper *, 1> result{nullptr};
            float result_dist = 1E99;
            auto nbResultsFound = tree.FindNeighbours(query_point, result.begin(), result.end(), result_dist);

            {
                const std::lock_guard<std::mutex> l(mutex);
                out << Diag::visited << ";" << Diag::minimizations << std::endl;
            }
        }));
        out.flush();
        out << "END BENCH" << std::endl << std::endl;
    }

    out.flush();
    out.close();
}

TEST(Benchmarks, DISABLED_NearestHit_LineSegments_Random)
{
    std::random_device dev{};

    std::fstream out(output_path+"NearestHit_LineSegments_Random.txt", std::fstream::out | std::fstream::app);
    if(out.fail())
    {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    //std::vector<unsigned int> bench_line_counts = {100, 250000, 500000, 750000, 1000000};
    std::vector<unsigned int> bench_line_counts = {100, 1000, 10000, 100000, 1000000};

    for(int bench_type_i = 0; bench_type_i < bench_line_counts.size(); ++bench_type_i)
    {
        std::cout << "Testing bench type " << (bench_type_i+1) << " of " << bench_line_counts.size() << std::endl;

        out << "BENCH INFO" << std::endl;
        unsigned int line_count = bench_line_counts[bench_type_i];
        unsigned int query_count = 100;
        float max_dist = 100;

        out << "Line count: " << line_count << std::endl;
        out << "Query count: " << query_count << std::endl;
        out << "Maximum distance: " << max_dist << std::endl;

        out << "START BENCH" << std::endl;
        std::mutex mutex;
        parallel_for(query_count, std::function([&out, &mutex, &dev, max_dist, query_count, line_count](unsigned int query_i){
            unsigned int query_seed, line_seed;
            {
                const std::lock_guard<std::mutex> l(mutex);
                query_seed = dev();
                line_seed = dev();
            }

            //Generate line, tree, query
            auto lines = GenerateRandomLineSegments(line_seed, line_count, max_dist, -100, 100);
            auto tree = segments::TreeBuilder<LineSegmentWrapper, &LineSegmentWrapper::l>::Build(lines.begin(), lines.end());
            auto query_point = GenerateRandomPoints(query_seed, 1, max_dist)[0];
            auto query_normal = GenerateRandomNormals(query_seed+1, 1)[0];

            //Perform query
            std::array<const LineSegmentWrapper *, 1> result{nullptr};
            float result_dist = 1E99;
            auto nbResultsFound = tree.FindNearestHits(query_point, query_normal, result.begin(), result.end(), result_dist);

            {
                const std::lock_guard<std::mutex> l(mutex);
                out << Diag::visited << ";" << Diag::minimizations << std::endl;
            }
        }));
        out.flush();

        out << "END BENCH" << std::endl << std::endl;
    }

    out.flush();
    out.close();
}

TEST(Benchmarks, DISABLED_NearestNeighbour_Short_LineSegments_Random)
{
    std::random_device dev{};

    std::fstream out(output_path+"NearestNeighbour_Short_LineSegments_Random.txt", std::fstream::out | std::fstream::app);
    if(out.fail())
    {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    //std::vector<unsigned int> bench_line_counts = {100, 250000, 500000, 750000, 1000000};
    //std::vector<unsigned int> bench_line_counts = {100, 1000, 10000, 100000, 1000000};
    std::vector<unsigned int> bench_line_counts = {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288};

    for(int bench_type_i = 0; bench_type_i < bench_line_counts.size(); ++bench_type_i)
    {
        std::cout << "Testing bench type " << (bench_type_i+1) << " of " << bench_line_counts.size() << std::endl;

        out << "BENCH INFO" << std::endl;
        unsigned int line_count = bench_line_counts[bench_type_i];
        unsigned int query_count = 100;
        float max_dist = 100;

        out << "Line count: " << line_count << std::endl;
        out << "Query count: " << query_count << std::endl;
        out << "Maximum distance: " << max_dist << std::endl;

        out << "START BENCH" << std::endl;
        std::mutex mutex;
        parallel_for(query_count, std::function([&out, &mutex, &dev, max_dist, query_count, line_count](unsigned int query_i){
            unsigned int query_seed, line_seed;
            {
                const std::lock_guard<std::mutex> l(mutex);
                query_seed = dev();
                line_seed = dev();
            }

            //Generate line, tree, query
            auto lines = GenerateRandomLineSegments(line_seed, line_count, max_dist, -100, 100);
            ModifyLineSegmentLength(lines, 0.1);
            auto tree = segments::TreeBuilder<LineSegmentWrapper, &LineSegmentWrapper::l>::Build(lines.begin(), lines.end());
            auto query_point = GenerateRandomPoints(query_seed, 1, max_dist)[0];

            //Perform query
            std::array<const LineSegmentWrapper *, 1> result{nullptr};
            float result_dist = 1E99;
            auto nbResultsFound = tree.FindNeighbours(query_point, result.begin(), result.end(), result_dist);

            {
                const std::lock_guard<std::mutex> l(mutex);
                out << Diag::visited << ";" << Diag::minimizations << std::endl;
            }
        }));
        out.flush();
        out << "END BENCH" << std::endl << std::endl;
    }

    out.flush();
    out.close();
}


TEST(Benchmarks, DISABLED_NearestNeighbour_Lines_SceneFile)
{
    std::random_device dev{};

    //"/media/wouter/Data2/Thesis/misc/sphere_lines.txt"
    std::string scene_filename = "/media/wouter/Data2/Thesis/misc/underwater_lines.txt";
    std::string scene_name = "sphere"; //sphere, underwater
    std::string filename;
    filename = output_path+"NearestNeighbour_Lines_Scene_"+scene_name+".txt";

    std::vector<unsigned int> bench_line_counts = {100, 1000, 10000, 100000, 1000000};

    std::fstream out(filename, std::fstream::out | std::fstream::app);
    if(out.fail())
    {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    std::vector<LineWrapper> all_lines = LoadFromFile(scene_filename);

    for(int bench_type_i = 0; bench_type_i < bench_line_counts.size(); ++bench_type_i)
    {
        std::cout << "Testing bench type " << (bench_type_i+1) << " of " << bench_line_counts.size() << std::endl;

        out << "BENCH INFO" << std::endl;
        unsigned int line_count = bench_line_counts[bench_type_i];
        unsigned int query_count = 100;
        float max_dist = 5;

        out << "Line count: " << line_count << std::endl;
        out << "Query count: " << query_count << std::endl;
        out << "Maximum distance: " << max_dist << std::endl;

        out << "START BENCH" << std::endl;
        std::mutex mutex;
        parallel_for(query_count, std::function([&out, &mutex, &dev, max_dist, query_count, &all_lines, line_count](unsigned int query_i){
            unsigned int query_seed, line_seed;
            {
                const std::lock_guard<std::mutex> l(mutex);
                query_seed = dev();
                line_seed = dev();
            }

            //Generate line, tree and query
            std::vector<LineWrapper> lines = SampleRandomLines(all_lines, line_seed, line_count);
            auto tree = TreeBuilder<LineWrapper, &LineWrapper::l>::Build(lines.begin(), lines.end());
            auto query_point = GenerateRandomPoints(query_seed, 1, max_dist)[0];

            //Perform query
            std::array<const LineWrapper *, 1> result{nullptr};
            float result_dist = 1E99;
            auto nbResultsFound = tree.FindNeighbours(query_point, result.begin(), result.end(), result_dist);

            {
                const std::lock_guard<std::mutex> l(mutex);
                out << Diag::visited << ";" << Diag::minimizations << std::endl;
            }
        }));
        out.flush();
        out << "END BENCH" << std::endl << std::endl;
    }

    out.flush();
    out.close();
}

TEST(Benchmarks, DISABLED_NearestNeighbour_Lines_Parallel)
{
    std::random_device dev{};

    std::fstream out(output_path+"NearestNeighbour_Lines_Parallel.txt", std::fstream::out | std::fstream::app);
    if(out.fail())
    {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    //std::vector<unsigned int> bench_line_counts = {100, 250000, 500000, 750000, 1000000};
    std::vector<unsigned int> bench_line_counts = {100, 1000, 10000, 100000, 1000000};

    for(int bench_type_i = 0; bench_type_i < bench_line_counts.size(); ++bench_type_i)
    {
        std::cout << "Testing bench type " << (bench_type_i+1) << " of " << bench_line_counts.size() << std::endl;

        out << "BENCH INFO" << std::endl;
        unsigned int line_count = bench_line_counts[bench_type_i];
        unsigned int query_count = 100;
        float max_dist = 100;

        out << "Line count: " << line_count << std::endl;
        out << "Query count: " << query_count << std::endl;
        out << "Maximum distance: " << max_dist << std::endl;

        out << "START BENCH" << std::endl;
        std::mutex mutex;
        parallel_for(query_count, std::function([&out, &mutex, &dev, max_dist, query_count, line_count](unsigned int query_i){
            unsigned int query_seed, line_seed;
            {
                const std::lock_guard<std::mutex> l(mutex);
                query_seed = dev();
                line_seed = dev();
            }

            //Generate line, tree, query
            std::default_random_engine rng{line_seed+1};
            std::uniform_real_distribution<float> dist(0, 1);
            Vector3f direction(dist(rng), dist(rng), dist(rng));
            direction.normalize();
            std::vector<LineWrapper> lines = GenerateParallelLines(line_seed, line_count, max_dist, direction);
            auto tree = TreeBuilder<LineWrapper, &LineWrapper::l>::Build(lines.begin(), lines.end());
            auto query_point = GenerateRandomPoints(query_seed, 1, max_dist)[0];

            //Perform query
            std::array<const LineWrapper *, 1> result{nullptr};
            float result_dist = 1E99;
            auto nbResultsFound = tree.FindNeighbours(query_point, result.begin(), result.end(), result_dist);

            {
                const std::lock_guard<std::mutex> l(mutex);
                out << Diag::visited << ";" << Diag::minimizations << std::endl;
            }
        }));
        out.flush();
        out << "END BENCH" << std::endl << std::endl;
    }

    out.flush();
    out.close();
}

TEST(Benchmarks, DISABLED_NearestNeighbour_Lines_EquiDistant)
{
    std::random_device dev{};

    std::fstream out(output_path+"NearestNeighbour_Lines_EquiDistant.txt", std::fstream::out | std::fstream::app);
    if(out.fail())
    {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    //std::vector<unsigned int> bench_line_counts = {100, 250000, 500000, 750000, 1000000};
    std::vector<unsigned int> bench_line_counts = {100, 1000, 10000, 100000, 1000000};

    for(int bench_type_i = 0; bench_type_i < bench_line_counts.size(); ++bench_type_i)
    {
        std::cout << "Testing bench type " << (bench_type_i+1) << " of " << bench_line_counts.size() << std::endl;

        out << "BENCH INFO" << std::endl;
        unsigned int line_count = bench_line_counts[bench_type_i];
        unsigned int query_count = 100;
        float max_dist = 100;

        out << "Line count: " << line_count << std::endl;
        out << "Query count: " << query_count << std::endl;
        out << "Maximum distance: " << max_dist << std::endl;

        out << "START BENCH" << std::endl;
        std::mutex mutex;
        parallel_for(query_count, std::function([&out, &mutex, &dev, max_dist, query_count, line_count](unsigned int query_i){
            unsigned int query_seed, line_seed;
            {
                const std::lock_guard<std::mutex> l(mutex);
                query_seed = dev();
                line_seed = dev();
            }

            //Generate lines and tree
            std::vector<LineWrapper> lines = GenerateEquiDistantLines(line_seed, line_count, max_dist);
            auto tree = TreeBuilder<LineWrapper, &LineWrapper::l>::Build(lines.begin(), lines.end());
            auto query_point = GenerateRandomPoints(query_seed, 1, max_dist)[0];

            //Perform query
            std::array<const LineWrapper *, 1> result{nullptr};
            float result_dist = 1E99;
            auto nbResultsFound = tree.FindNeighbours(query_point, result.begin(), result.end(), result_dist);

            {
                const std::lock_guard<std::mutex> l(mutex);
                out << Diag::visited << ";" << Diag::minimizations << std::endl;
            }
        }));
        out.flush();
        out << "END BENCH" << std::endl << std::endl;
    }

    out.flush();
    out.close();
}

TEST(Benchmarks, DISABLED_NearestNeighbour_Lines_EqualMoment)
{
    std::random_device dev{};

    std::fstream out(output_path+"NearestNeighbour_Lines_EqualMoment.txt", std::fstream::out | std::fstream::app);
    if(out.fail())
    {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    //std::vector<unsigned int> bench_line_counts = {100, 250000, 500000, 750000, 1000000};
    std::vector<unsigned int> bench_line_counts = {100, 1000, 10000, 100000, 1000000};

    for(int bench_type_i = 0; bench_type_i < bench_line_counts.size(); ++bench_type_i)
    {
        std::cout << "Testing bench type " << (bench_type_i+1) << " of " << bench_line_counts.size() << std::endl;

        out << "BENCH INFO" << std::endl;
        unsigned int line_count = bench_line_counts[bench_type_i];
        unsigned int query_count = 100;
        float max_dist = 100;

        out << "Line count: " << line_count << std::endl;
        out << "Query count: " << query_count << std::endl;
        out << "Maximum distance: " << max_dist << std::endl;

        out << "START BENCH" << std::endl;

        std::mutex mutex;
        parallel_for(query_count, std::function([&out, &mutex, &dev, max_dist, query_count, line_count](unsigned int query_i){
            unsigned int query_seed, line_seed;
            {
                const std::lock_guard<std::mutex> l(mutex);
                query_seed = dev();
                line_seed = dev();
            }

            //Generate lines and tree
            std::default_random_engine rng{line_seed+1};
            std::uniform_real_distribution<float> dist(0, 1);
            Vector3f moment(dist(rng), dist(rng), dist(rng));
            moment.normalize();
            moment *= dist(rng) * max_dist;
            std::vector<LineWrapper> lines = GenerateEqualMomentLines(line_seed, line_count, moment);
            auto tree = TreeBuilder<LineWrapper, &LineWrapper::l>::Build(lines.begin(), lines.end());
            auto query_point = GenerateRandomPoints(query_seed, 1, max_dist)[0];

            //Perform query
            std::array<const LineWrapper *, 1> result{nullptr};
            float result_dist = 1E99;
            auto nbResultsFound = tree.FindNeighbours(query_point, result.begin(), result.end(), result_dist);

            {
                const std::lock_guard<std::mutex> l(mutex);
                out << Diag::visited << ";" << Diag::minimizations << std::endl;
            }
        }));
        out.flush();
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
        /*for(auto& s : lines)
        {
            s.l.t1 = t_min;
            s.l.t2 = t_max;
        }*/
        ModifyLineSegmentLength(lines, t_max-t_min);

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

TEST(Benchmarks, NearestNeighbour_Lines_Origin)
{
    std::fstream out(output_path+"NearestNeighbour_Lines_Origin.txt", std::fstream::out | std::fstream::app);
    if(out.fail())
    {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    unsigned int line_count = 100000;
    unsigned int iterations = 10;
    unsigned int query_count = 100;
    unsigned int query_seed, line_seed;
    {
        std::random_device dev{};
        query_seed = dev();
        line_seed = dev();
    }

    // Generate lines
    float max_line_dist = 10;
    std::vector<std::vector<LineWrapper>> lineSets(iterations);
    for(unsigned int i = 0; i < iterations; ++i)
    {
        lineSets[i] = GenerateRandomLines(line_seed+i, line_count, max_line_dist);
    }
    std::vector<std::vector<LineWrapper>> translatedLinesSets = lineSets;

    //Generate query points
    float max_query_dist = 100;
    auto query_points = GenerateRandomPoints(query_seed, query_count, max_query_dist);
    std::vector<Vector3f> translatedQueryPoints = query_points;

    //Generate translation vectors
    std::vector<Vector3f> bench_translation_vectors {
        Vector3f(0, 0, 0),
        Vector3f(20, 0, 0),
        Vector3f(40, 0, 0),
        Vector3f(60, 0, 0),
        Vector3f(0, 0, 20),
        Vector3f(0, 0, 40),
        Vector3f(0, 0, 60),
        Vector3f(0, 60, 0),
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

        for(unsigned int it = 0; it < iterations; ++it)
        {
            for(unsigned int i = 0; i < line_count; ++i)
            {
                Vector3f curP = lineSets[it][i].l.d.cross(lineSets[it][i].l.m);
                Vector3f p1 = curP + translation_vect;
                Vector3f p2 = (curP + lineSets[it][i].l.d) + translation_vect;
                translatedLinesSets[it][i] = LineWrapper(Line::FromTwoPoints(p1, p2));
            }
        }
        for(unsigned int i = 0; i < query_count; ++i)
        {
            translatedQueryPoints[i] = query_points[i] + translation_vect;
        }

        /*out << "START TREEDEPTH DATA" << std::endl;
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
        out << "END TREEDEPTH DATA" << std::endl;*/

        out << "START BENCH" << std::endl;

        out << "Query generation seed: " << query_seed << std::endl;
        out << "Line generation seed: " << line_seed << std::endl;


        //Perform queries
        out << "START DATA" << std::endl;
        std::cout << "Running queries        \r";
        for(unsigned int it = 0; it < iterations; ++it)
        {
            auto tree = TreeBuilder<LineWrapper, &LineWrapper::l>::Build(translatedLinesSets[it].begin(), translatedLinesSets[it].end());

            std::mutex mutex;
            parallel_for(translatedQueryPoints.size(), std::function([&translatedQueryPoints, &tree, &out, &mutex](std::vector<Vector3f>::size_type query_i){
                const auto& query = translatedQueryPoints[query_i];
                std::array<const LineWrapper *, 1> result {nullptr};
                float result_dist = 1E99;
                auto nbResultsFound = tree.FindNeighbours(query, result.begin(), result.end(), result_dist);

                {
                    const std::lock_guard<std::mutex> l(mutex);
                    out << Diag::visited << ";" << Diag::minimizations << std::endl;
                }
            }));
        }
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
    unsigned int iterations = 1;
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
        //auto lines = LoadFromFile("/home/wouter/Documents/sphere_lines.txt");


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

TEST(Benchmarks, DISABLED_NearestNeighbour_Lines_Random_SplitTypes)
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
        unsigned int iteration_count = 10;
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
                if(node == nullptr || (node->children[0] == nullptr || node->children[1] == nullptr))
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

TEST(Benchmarks, DISABLED_NearestNeighbour_Lines_Random_SplitTypes_By_Level)
{
    std::fstream out(output_path+"NearestNeighbour_Lines_Random_SplitTypes_By_Level.txt", std::fstream::out | std::fstream::app);
    if(out.fail())
    {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    unsigned int line_count = 100000;

    {
        out << "BENCH INFO" << std::endl;
        unsigned int iteration_count = 1;
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
            //auto lines = LoadFromFile("/home/wouter/Documents/sphere_lines.txt");
            std::vector<LineWrapper> lines = GenerateRandomLines(line_seed, line_count, max_dist);
            auto tree = TreeBuilder<LineWrapper, &LineWrapper::l>::Build(lines.begin(), lines.end());

            //Perform queries
            out << "START DATA" << std::endl;
            std::cout << "Running queries        \r";

            Eigen::Matrix<int, 4, 100> split_counts = Eigen::Matrix<int, 4, 100>::Zero(); // Rows are {phi, theta, radius, directional}, columns are levels
            std::function<void(const TreeNode<LineWrapper, &LineWrapper::l>*, int)> visitor;
            visitor = std::function([&visitor, &split_counts](const TreeNode<LineWrapper, &LineWrapper::l>* node, int level){
                if(node == nullptr || (node->children[0] == nullptr || node->children[1] == nullptr))
                {
                    return;
                }

                if(node->type == NodeType::moment)
                {
                    split_counts(node->bound_component_idx, level)++;
                }else if(node->type == NodeType::direction)
                {
                    split_counts(3, level)++;
                }

                visitor(node->children[0].get(), level+1);
                visitor(node->children[1].get(), level+1);
            });
            for(const auto& sector : tree.sectors)
            {
                visitor(sector.rootNode.get(), 0);
            }

            Eigen::Matrix<float, 4, 100> split_freqs;
            for(int level = 0; level < 100; ++level)
            {
                unsigned int total = split_counts.col(level).sum();
                split_freqs.col(level) = split_counts.col(level).cast<float>() / (float)total;
            }

            for(int level = 0; level < 100; ++level)
            {
                for(int dim_i = 0; dim_i < 4; ++dim_i)
                {
                    auto curVal = split_freqs(dim_i, level);
                    curVal = std::isnan(curVal) ? 0 : curVal;
                    out << curVal;
                    if(dim_i < 3)
                    {
                        out << ";";
                    }else{
                        out << std::endl;
                    }
                }
            }
            std::cout << "                       \r";
            out << "END DATA" << std::endl;
            out.flush();
        }
        out << "END BENCH" << std::endl << std::endl;
    }

    out.flush();
    out.close();
}

TEST(Benchmarks, DISABLED_NearestNeighbour_Lines_Random_Build_Variances)
{
    std::fstream out(output_path+"NearestNeighbour_Lines_Random_Build_Variances.txt", std::fstream::out | std::fstream::app);
    if(out.fail())
    {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    unsigned int line_count = 100000;

    {
        out << "BENCH INFO" << std::endl;
        unsigned int iteration_count = 1;
        float max_dist = 100;

        out << "Line count: " << line_count << std::endl;
        out << "Iteration count: " << iteration_count << std::endl;
        out << "Maximum distance: " << max_dist << std::endl;

        Diag::on_build_variance_calculated = [&out](float m_phi_var, float m_gamma_var, float m_radius_var, float d_var){
            out << m_phi_var << ";" << m_gamma_var << ";" << m_radius_var << ";" << d_var << std::endl;
        };

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

            out << "START DATA" << std::endl;
            auto tree = TreeBuilder<LineWrapper, &LineWrapper::l>::Build(lines.begin(), lines.end());
            out << "END DATA" << std::endl;
            out.flush();
        }
        out << "END BENCH" << std::endl << std::endl;
    }

    out.flush();
    out.close();

    Diag::reset();
}

TEST(Benchmarks, DISABLED_NearestNeighbour_Lines_Random_Spreads)
{
    std::fstream out(output_path+"NearestNeighbour_Lines_Random_Spreads.txt", std::fstream::out | std::fstream::app);
    if(out.fail())
    {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    unsigned int line_count = 1000000;

    {
        out << "BENCH INFO" << std::endl;
        unsigned int iteration_count = 10;
        float max_dist = 100;

        out << "Line count: " << line_count << std::endl;
        out << "Iteration count: " << iteration_count << std::endl;
        out << "Maximum distance: " << max_dist << std::endl;

        out << "START BENCH" << std::endl;
        unsigned int line_seed;
        {
            std::random_device dev{};
            line_seed = dev();
        }

        out << "Line generation seed: " << line_seed << std::endl;

        auto lines = LoadFromFile("/home/wouter/Documents/sphere_lines.txt");
        for(auto& l : lines)
        {
            l.l.m.z() *= 2.0;
        }
        //std::vector<LineWrapper> lines = GenerateRandomLines(line_seed, line_count, max_dist);

        out << "START DATA" << std::endl;

        for(const auto& line : lines)
        {
            auto m = cart2spherical(line.l.m);
            out << m.x() << ";" << m.y() << ";" << m.z() << std::endl;
        }

        out << "END DATA" << std::endl;

        out << "END BENCH" << std::endl << std::endl;
    }

    out.flush();
    out.close();
}
