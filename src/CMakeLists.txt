# Installation instructions
#install(TARGETS my_library DESTINATION "${main_lib_dest}")
#install(FILES ${header} DESTINATION "${include_dest}")

file(GLOB_RECURSE PLUCKERTREE_SRC
    "*.h"
    "*.cpp"
)

add_library(pluckertree ${PLUCKERTREE_SRC})

target_include_directories(pluckertree PRIVATE ${PLUCKERTREE_SOURCE_DIR}/src)
target_include_directories(pluckertree PRIVATE ${PLUCKERTREE_SOURCE_DIR}/include)
target_include_directories(pluckertree PRIVATE ${PLUCKERTREE_SOURCE_DIR}/dependencies/LBFGSpp/include)

find_package(Eigen3 REQUIRED NO_MODULE)
target_link_libraries(pluckertree PRIVATE Eigen3::Eigen nlopt)