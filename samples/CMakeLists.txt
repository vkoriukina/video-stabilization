set(target "stabilizer")

file(GLOB hdrs "*.hpp")
file(GLOB srcs "*.cpp")

add_executable(${target} ${srcs} ${hdrs})
target_link_libraries(${target} ${LIBRARY_DEPS})
