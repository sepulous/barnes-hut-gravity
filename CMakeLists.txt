cmake_minimum_required(VERSION 3.25)
project(BarnesHut LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  #set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_subdirectory("third_party/glfw-3.5.1")

file(GLOB IMGUI_SOURCES
	"third_party/imgui-1.92.9b/*.cpp"
	"third_party/imgui-1.92.9b/backends/imgui_impl_glfw.cpp"
	"third_party/imgui-1.92.9b/backends/imgui_impl_opengl3.cpp"
)

add_executable(BarnesHut
	"src/main.cpp"
	"src/quad_tree.cpp"
	"src/shader_program.cpp"

	${IMGUI_SOURCES}
	"third_party/glad/src/glad.c"
)

target_link_libraries(BarnesHut PRIVATE glfw OpenMP::OpenMP_CXX CUDA::cudart)

target_include_directories(BarnesHut PRIVATE
	"third_party/glfw-3.5.1/include"
	"third_party/imgui-1.92.9b"
	"third_party/imgui-1.92.9b/backends"
	"third_party/glad/include"
	"third_party/glm"
)
