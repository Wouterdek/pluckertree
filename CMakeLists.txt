cmake_minimum_required (VERSION 3.10)
project(PLUCKERTREE)

set(CMAKE_CXX_STANDARD 17)

#set(CMAKE_VERBOSE_MAKEFILE ON)

add_subdirectory (src)

if(DEFINED PLUCKERTREE_ENABLE_TESTS)
  add_subdirectory (tests)
endif()
