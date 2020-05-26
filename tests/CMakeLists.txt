enable_testing()

file(GLOB_RECURSE tests_SRC
        "*.h"
        "*.cpp"
        )

### GOOGLE TEST
if(NOT EXISTS "${PLUCKERTREE_SOURCE_DIR}/dependencies/googletest/CMakeLists.txt")
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory("${PLUCKERTREE_SOURCE_DIR}/dependencies/googletest/" "dependencies/googletest/")

# Keeps the cache cleaner
mark_as_advanced(
        BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
        gmock_build_tests gtest_build_samples gtest_build_tests
        gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
)
set_target_properties(gtest PROPERTIES FOLDER dependencies)
set_target_properties(gtest_main PROPERTIES FOLDER dependencies)
set_target_properties(gmock PROPERTIES FOLDER dependencies)
set_target_properties(gmock_main PROPERTIES FOLDER dependencies)
###

find_package(Eigen3 REQUIRED NO_MODULE)
#find_package(Boost 1.69 REQUIRED COMPONENTS fiber)
#find_package(TBB CONFIG REQUIRED)

add_executable(tests ${tests_SRC})

target_include_directories(tests PRIVATE ${PLUCKERTREE_SOURCE_DIR}/include)
target_include_directories(tests PRIVATE ${PLUCKERTREE_SOURCE_DIR}/src)
target_link_libraries (tests PRIVATE gmock gtest gtest_main pluckertree Eigen3::Eigen)

#find_package(GTest MODULE REQUIRED)

#gtest_add_tests(TARGET tests tests_SRC)
#gtest_discover_tests(tests)
