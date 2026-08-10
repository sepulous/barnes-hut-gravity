#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <chrono>
#include <type_traits>
#include <span>
#include <cstdint>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "particle.h"
#include "shaders.h"
#include "shader_program.h"
#include "quad_tree.h"
#include "cuda.cuh"

#define RENDER 1
#define PARALLEL 1
#define MAX_STEPS 0

constexpr size_t NUM_PARTICLES = 10'000;
constexpr double TIME_STEP = 0.01;
constexpr double TIME_STEP_SQUARED = TIME_STEP * TIME_STEP;
constexpr double THETA = 0.8; // width/distance threshold for quad tree cells
constexpr double SOFTENING = 1e-3;
constexpr double METERS_PER_UNIT = 1.0;
constexpr double G = 6.67430e-11 / (METERS_PER_UNIT * METERS_PER_UNIT);
constexpr size_t LEAF_CAPACITY = 64;
constexpr size_t MAX_TREE_DEPTH = 8;

double rand_range(double min_inclusive, double max_inclusive);
glm::dvec2 GetAccelerationAtParticle(Particle* point, QuadTree& tree);

int main()
{
	//
	// Set up GLFW window
	//
	GLFWwindow* window;

	if (!glfwInit())
		return -1;

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(800, 800, "Barnes-Hut Gravity Simulation", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	glfwSwapInterval(0);

	//
	// Set up GLAD
	//
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cerr << "Failed to initialize GLAD\n";
		return -1;
	}

	//
	// Set up ImGUI
	//
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 460 core");

	//
	// Set up OpenGL
	//
	glEnable(GL_PROGRAM_POINT_SIZE);

	GLuint particle_vbo;
	GLuint particle_vao;

	glGenVertexArrays(1, &particle_vao);
	glBindVertexArray(particle_vao);

	glCreateBuffers(1, &particle_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, particle_vbo);

	glNamedBufferStorage(
		particle_vbo,
		NUM_PARTICLES * (2 * sizeof(float)),
		nullptr,
		GL_MAP_WRITE_BIT |
		GL_MAP_PERSISTENT_BIT |
		GL_MAP_COHERENT_BIT
	);

	float* gpu_positions = static_cast<float*>(
		glMapNamedBufferRange(
			particle_vbo,
			0,
			NUM_PARTICLES * (2 * sizeof(float)),
			GL_MAP_WRITE_BIT |
			GL_MAP_PERSISTENT_BIT |
			GL_MAP_COHERENT_BIT
		)
	);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
	glEnableVertexAttribArray(0);

	ShaderProgram shader_program(POINT_VERTEX_SHADER, POINT_FRAGMENT_SHADER);

	//
	// Configuration
	//
	std::vector<Particle> particles;
	particles.reserve(NUM_PARTICLES);
	for (int i = 0; i < NUM_PARTICLES; i++)
	{
		particles.push_back({
			.position = {
				rand_range(-1.0, 1.0),
				rand_range(-1.0, 1.0)
			},
			.velocity = {0.0, 0.0},
			.acceleration = {0.0, 0.0},
			.mass = rand_range(1.0, 50.0)
		});
		particles[i].next_position = particles[i].position;
	}

	float* new_accelerations = new float[2 * NUM_PARTICLES];

	QuadTree::SetMaxDepth(MAX_TREE_DEPTH);
	QuadTree::SetLeafCapacity(LEAF_CAPACITY);

	CUDAContext cuda_context(NUM_PARTICLES, MAX_TREE_DEPTH);

	long long construction_time = 0;
	long long mass_time = 0;
	long long acceleration_compute_time = 0;
	long long integration_time = 0;
	long long render_time = 0;

	//
	// Simulation
	//
	#if MAX_STEPS
	int step = 0;
	#endif
	double start_time = glfwGetTime();
	while (!glfwWindowShouldClose(window))
	{
		#if RENDER
		glfwPollEvents();
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		#endif

		QuadTree::GetPool().clear();

		//
		// Update positions and construct quad tree
		//
		auto const_time_start = std::chrono::steady_clock::now();
		for (int i = 0; i < NUM_PARTICLES; i++)
		{
			Particle& particle = particles[i];
			particle.position = particle.next_position;
			#if RENDER
			gpu_positions[2*i] = static_cast<float>(particle.position.x);
			gpu_positions[2*i + 1] = static_cast<float>(particle.position.y);
			#endif
		}
		QuadTree::GetPool().emplace_back(glm::dvec2{ 0, 0 }, glm::dvec2{ 1.1, 1.1 }); // TODO: Adjust size based on particle positions
		QuadTree::GetPool()[0].Build(std::span<Particle>(particles.begin(), particles.end()));
		auto mass_time_start = std::chrono::steady_clock::now();
		QuadTree::GetPool()[0].CalculateMass();
		auto const_time_end = std::chrono::steady_clock::now();
		construction_time += (const_time_end - const_time_start).count();
		mass_time += (const_time_end - mass_time_start).count();

		//
		// Integrate
		//
		auto accel_time_start = std::chrono::steady_clock::now();
		ComputeAccelerations(cuda_context, particles, QuadTree::GetPool(), new_accelerations); // Blocks while GPU finishes
		auto accel_time_end = std::chrono::steady_clock::now();
		acceleration_compute_time += (accel_time_end - accel_time_start).count();

		auto int_time_start = std::chrono::steady_clock::now();
		#pragma omp parallel for
		for (int i = 0; i < NUM_PARTICLES; i++)
		{
			glm::dvec2 new_acceleration{ new_accelerations[2*i], new_accelerations[2*i + 1] };
			Particle& particle = particles[i];
			particle.next_position = particle.position + TIME_STEP*particle.velocity + 0.5*TIME_STEP_SQUARED*particle.acceleration;
			particle.velocity = particle.velocity + 0.5 * (particle.acceleration + new_acceleration) * TIME_STEP;
			particle.acceleration = new_acceleration;
		}
		auto int_time_end = std::chrono::steady_clock::now();
		integration_time += (int_time_end - int_time_start).count();

		#if MAX_STEPS
		step++;
		//printf("Step: %i\n", step);
		if (step == MAX_STEPS)
		{
			printf("Computed %i steps with %i particles in %f seconds\n", MAX_STEPS, NUM_PARTICLES, glfwGetTime() - start_time);
			break;
		}
		#endif

		//
		// Render
		//
		#if RENDER
		auto render_time_start = std::chrono::steady_clock::now();
		glClear(GL_COLOR_BUFFER_BIT);

		shader_program.Use();
		glBindVertexArray(particle_vao);
		glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		glfwSwapBuffers(window);
		auto render_time_end = std::chrono::steady_clock::now();
		render_time += (render_time_end - render_time_start).count();
		#endif
	}

	double total_time = static_cast<double>(construction_time + integration_time + acceleration_compute_time + render_time);
	std::cout << "Construction: " << (construction_time / 1'000'000) << "ms (" << (100.0 * static_cast<double>(construction_time) / total_time) << "%)" << std::endl;
	std::cout << "    Mass Calculation: " << (mass_time / 1'000'000) << "ms (" << (100.0 * static_cast<double>(mass_time) / static_cast<double>(construction_time)) << "% of construction)" << std::endl;
	std::cout << "Acceleration Computation: " << (acceleration_compute_time / 1'000'000) << "ms (" << (100.0 * static_cast<double>(acceleration_compute_time) / total_time) << "%)" << std::endl;
	std::cout << "Integration: " << (integration_time / 1'000'000) << "ms (" << (100.0 * static_cast<double>(integration_time) / total_time) << "%)" << std::endl;
	std::cout << "Rendering: " << (render_time / 1'000'000) << "ms (" << (100.0 * static_cast<double>(render_time) / total_time) << "%)" << std::endl;

	std::cin.get();

	glfwTerminate();
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	return 0;
}

double rand_range(double min_inclusive, double max_inclusive)
{
	//static std::mt19937_64 generator{ std::random_device{}() };
	static std::mt19937_64 generator{ 69420 };
	std::uniform_real_distribution<double> dist(min_inclusive, max_inclusive);
	return dist(generator);
}

glm::dvec2 GetAccelerationAtParticle(Particle* point, QuadTree& root)
{
	glm::dvec2 acceleration{ 0 };

	std::vector<QuadTree*> stack{ &root };
	
	while (!stack.empty())
	{
		auto tree = stack.back();
		stack.pop_back();

		if (!tree->HasChildren())
		{
			for (auto& particle : tree->GetParticles())
			{
				if (&particle != point)
				{
					auto displacement = particle.position - point->position;
					double distance_squared = glm::dot(displacement, displacement);
					double soft_inv_distance = distance_squared + SOFTENING * SOFTENING;
					soft_inv_distance *= soft_inv_distance * soft_inv_distance;
					soft_inv_distance = 1.0 / std::sqrt(soft_inv_distance);
					acceleration += (G * particle.mass * soft_inv_distance) * displacement;
				}
			}
		}
		else
		{
			glm::dvec2 com_displacement = tree->GetCenterOfMass() - point->position;
			double com_distance_squared = glm::dot(com_displacement, com_displacement);
			double width = 2.0 * tree->GetExtents().x;

			if (width * width < com_distance_squared * THETA * THETA)
			{
				double soft_inv_distance = com_distance_squared + SOFTENING * SOFTENING;
				soft_inv_distance *= soft_inv_distance * soft_inv_distance;
				soft_inv_distance = 1.0 / std::sqrt(soft_inv_distance);
				acceleration += (G * tree->GetTotalMass() * soft_inv_distance) * com_displacement;
			}
			else
			{
				//for (auto child_index : tree->GetChildren())
				//	acceleration += GetAccelerationAtParticle(point, QuadTree::GetPool()[child_index]);

				for (auto child_index : tree->GetChildren())
					stack.push_back(&QuadTree::GetPool()[child_index]);
			}
		}
	}

	return acceleration;
}
