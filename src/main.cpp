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

#include "simulation_settings.h"
#include "particle.h"
#include "renderer.h"
#include "quad_tree.h"
#include "cuda.cuh"

#define RENDER 1
#define PARALLEL 1
#define MAX_STEPS 0

double rand_range(double min_inclusive, double max_inclusive);

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
	// Configuration
	//

	SimulationSettings settings{
		.particle_count = 10'000,
		.leaf_capacity = 64,
		.maximum_tree_depth = 8,
		.threads_per_block = 256,
		.softening = 1e-3,
		.theta = 0.8,
		.time_step = 0.01
	};

	std::vector<Particle> particles;
	particles.reserve(settings.particle_count);
	for (int i = 0; i < settings.particle_count; i++)
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

	//
	// Simulation
	//

	Renderer::Init(settings.particle_count);

	float* new_accelerations = new float[2 * settings.particle_count];

	QuadTree::SetMaxDepth(settings.maximum_tree_depth);
	QuadTree::SetLeafCapacity(settings.leaf_capacity);

	CUDAContext cuda_context(settings.particle_count, settings.maximum_tree_depth);

	long long construction_time = 0;
	long long mass_time = 0;
	long long acceleration_compute_time = 0;
	long long integration_time = 0;
	long long render_time = 0;

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
		for (int i = 0; i < settings.particle_count; i++)
		{
			Particle& particle = particles[i];
			particle.position = particle.next_position;
			#if RENDER
			Renderer::SetParticlePosition(i, particle.position);
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
		ComputeAccelerations(cuda_context, particles, QuadTree::GetPool(), new_accelerations, settings.threads_per_block); // Blocks while GPU finishes
		auto accel_time_end = std::chrono::steady_clock::now();
		acceleration_compute_time += (accel_time_end - accel_time_start).count();

		auto int_time_start = std::chrono::steady_clock::now();
		#pragma omp parallel for
		for (int i = 0; i < settings.particle_count; i++)
		{
			glm::dvec2 new_acceleration{ new_accelerations[2*i], new_accelerations[2*i + 1] };
			Particle& particle = particles[i];
			particle.next_position = particle.position + settings.time_step * particle.velocity + 0.5 * settings.time_step * settings.time_step * particle.acceleration;
			particle.velocity = particle.velocity + 0.5 * (particle.acceleration + new_acceleration) * settings.time_step;
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

		Renderer::Render();

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
