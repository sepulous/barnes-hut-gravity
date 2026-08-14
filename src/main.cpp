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
#include "gpu_info.h"
#include "cuda.cuh"

float zoom = 1.0f;

float rand_range(float min_inclusive, float max_inclusive);
void ComputeAccelerationsCPU(const std::vector<Particle>& particles, const std::vector<QuadTree>& tree, float* new_accelerations, SimulationSettings settings);

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

	window = glfwCreateWindow(1280, 720, "Barnes-Hut Gravity Simulation", NULL, NULL);
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

	GPUInfo gpu_info = GetGPUInfo();

	const uint32_t leaf_capacity_step = 8;
	const uint32_t maximum_tree_depth_step = 1;
	const uint32_t threads_per_block_step = gpu_info.warp_size;
	const float theta_min = 0;
	const float theta_max = 2.0;

	bool running = false;
	bool reset = true;

	glfwSetScrollCallback(window, [](GLFWwindow* window, double scroll_x, double scroll_y) {
		int scroll = static_cast<int>(scroll_y);
		if (scroll > 0)
			zoom *= 1.1f;
		else
			zoom /= 1.1f;
		
		Renderer::SetZoom(zoom);
	});

	SimulationSettings settings {
		.max_steps = 1000,
		.particle_count = 4'000,
		.leaf_capacity = 64,
		.maximum_tree_depth = 8,
		.threads_per_block = 256,
		.softening = 1e-6f,
		.theta = 0.5f,
		.time_step = 0.01f
	};

	std::vector<Particle> initial_configuration;
	initial_configuration.reserve(settings.particle_count);

	std::vector<Particle> particles;
	particles.reserve(settings.particle_count);

	for (uint64_t i = 0; i < settings.particle_count; i++)
	{
		particles.push_back({
			.position = {
				rand_range(-1.0f, 1.0f),
				rand_range(-1.0f, 1.0f)
			},
			.velocity = {0.0f, 0.0f},
			.acceleration = {0.0f, 0.0f},
			.mass = rand_range(1.0f, 50.0f)
		});
		particles[i].next_position = particles[i].position;
	}
	initial_configuration = particles;

	//
	// Simulation
	//

	CUDAContext cuda_context;
	if (gpu_info.cuda_supported)
		cuda_context = CUDAContext(settings.particle_count, settings.maximum_tree_depth);

	Renderer::Init(settings.particle_count);
	Renderer::SetZoom(1.0f);

	float* new_accelerations = new float[2 * settings.particle_count];

	long long construction_time = 0;
	long long mass_time = 0;
	long long acceleration_compute_time = 0;
	long long integration_time = 0;
	long long render_time = 0;

	float max_position_coord = 1.0f;

	unsigned step = 0;
	double start_time = glfwGetTime();
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		if (reset)
		{
			particles = initial_configuration;
			for (int i = 0; i < settings.particle_count; i++)
				Renderer::SetParticlePosition(i, particles[i].position);

			reset = false;
		}

		if (running)
		{
			QuadTree::GetPool().clear();
			QuadTree::SetMaxDepth(settings.maximum_tree_depth);
			QuadTree::SetLeafCapacity(settings.leaf_capacity);

			//
			// Update positions and construct quad tree
			//

			auto const_time_start = std::chrono::steady_clock::now();
			for (int i = 0; i < settings.particle_count; i++)
			{
				Particle& particle = particles[i];
				particle.position = particle.next_position;
				Renderer::SetParticlePosition(i, particle.position);

				max_position_coord = glm::max(max_position_coord, glm::abs(particle.position.x));
				max_position_coord = glm::max(max_position_coord, glm::abs(particle.position.y));
			}
			QuadTree::GetPool().emplace_back(glm::dvec2{ 0, 0 }, glm::dvec2{ 1.1 * max_position_coord, 1.1 * max_position_coord }); // TODO: Adjust size based on particle positions
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
			if (gpu_info.cuda_supported)
				ComputeAccelerationsCUDA(cuda_context, particles, QuadTree::GetPool(), new_accelerations, settings); // Blocks while GPU finishes
			else
				ComputeAccelerationsCPU(particles, QuadTree::GetPool(), new_accelerations, settings);
			auto accel_time_end = std::chrono::steady_clock::now();
			acceleration_compute_time += (accel_time_end - accel_time_start).count();

			auto int_time_start = std::chrono::steady_clock::now();
			#pragma omp parallel for
			for (int i = 0; i < settings.particle_count; i++)
			{
				glm::vec2 new_acceleration{ new_accelerations[2 * i], new_accelerations[2 * i + 1] };
				Particle& particle = particles[i];
				particle.next_position = particle.position + settings.time_step * particle.velocity + 0.5f * settings.time_step * settings.time_step * particle.acceleration;
				particle.velocity += 0.5f * (particle.acceleration + new_acceleration) * settings.time_step;
				particle.acceleration = new_acceleration;
			}
			auto int_time_end = std::chrono::steady_clock::now();
			integration_time += (int_time_end - int_time_start).count();

			step++;
			if (settings.max_steps > 0 && step >= settings.max_steps)
			{
				running = false;
				printf("Computed %i steps with %i particles in %f seconds\n", settings.max_steps, settings.particle_count, glfwGetTime() - start_time);
			}
		}

		//
		// Render
		//

		auto render_time_start = std::chrono::steady_clock::now();

		int control_width = static_cast<int>(0.25f * io.DisplaySize.x);
		int view_width = static_cast<int>(io.DisplaySize.x) - control_width;

		ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
		ImGui::SetNextWindowSize(ImVec2(control_width, io.DisplaySize.y), ImGuiCond_Once);

		ImGui::Begin("Controls", 0, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove );

		if (running)
			ImGui::BeginDisabled();

		ImGui::InputScalar("Max Steps", ImGuiDataType_U32, &settings.max_steps);
		ImGui::InputScalar("Leaf Capacity", ImGuiDataType_U32, &settings.leaf_capacity, &leaf_capacity_step, &leaf_capacity_step);
		ImGui::InputScalar("Maximum Tree Depth", ImGuiDataType_U32, &settings.maximum_tree_depth, &maximum_tree_depth_step, &maximum_tree_depth_step);
		if (gpu_info.cuda_supported)
			ImGui::InputScalar("Threads/Block", ImGuiDataType_U32, &settings.threads_per_block, &threads_per_block_step, &threads_per_block_step);
		ImGui::InputFloat("Softening", &settings.softening, 0, 0, "%.6f");
		ImGui::InputFloat("Theta", &settings.theta, 0.0f, 2.0f, "%.2f");
		ImGui::InputFloat("Time Step", &settings.time_step, 0, 0, "%.6f");

		if (running)
			ImGui::EndDisabled();

		if (!running)
		{
			if (ImGui::Button("Start"))
				running = true;

			ImGui::SameLine();

			if (ImGui::Button("Reset"))
			{
				reset = true;
				step = 0;
			}
		}
		else
		{
			if (ImGui::Button("Pause"))
				running = false;

			ImGui::SameLine();

			ImGui::BeginDisabled();
			ImGui::Button("Reset");
			ImGui::EndDisabled();
		}

		control_width = static_cast<int>(ImGui::GetContentRegionAvail().x + 2 * ImGui::GetStyle().WindowPadding.x);
		view_width = static_cast<int>(io.DisplaySize.x) - control_width;

		ImGui::End();

		ImGui::SetNextWindowPos(ImVec2(control_width, 0), ImGuiCond_Always);
		ImGui::SetNextWindowSize(ImVec2(view_width, io.DisplaySize.y), ImGuiCond_Always);

		ImGui::Begin("Visualization", 0, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar);

		auto avail = ImGui::GetContentRegionAvail();
		float size = glm::max(avail.x, avail.y);

		Renderer::Resize(avail.x, avail.y);
		Renderer::Render();

		ImGui::Image(
			(ImTextureID)Renderer::GetTexture(),
			ImVec2(size, size)
		);

		ImGui::End();

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		auto render_time_end = std::chrono::steady_clock::now();
		if (running)
			render_time += (render_time_end - render_time_start).count();

		glfwSwapBuffers(window);
	}

	double total_time = static_cast<double>(construction_time + integration_time + acceleration_compute_time + render_time);
	std::cout << "Construction: " << (construction_time / 1'000'000) << "ms (" << (100.0 * static_cast<double>(construction_time) / total_time) << "%)" << std::endl;
	std::cout << "    Mass Calculation: " << (mass_time / 1'000'000) << "ms (" << (100.0 * static_cast<double>(mass_time) / static_cast<double>(construction_time)) << "% of construction)" << std::endl;
	std::cout << "Acceleration Computation: " << (acceleration_compute_time / 1'000'000) << "ms (" << (100.0 * static_cast<double>(acceleration_compute_time) / total_time) << "%)" << std::endl;
	std::cout << "Integration: " << (integration_time / 1'000'000) << "ms (" << (100.0 * static_cast<double>(integration_time) / total_time) << "%)" << std::endl;
	std::cout << "Rendering: " << (render_time / 1'000'000) << "ms (" << (100.0 * static_cast<double>(render_time) / total_time) << "%)" << std::endl;

	glfwTerminate();
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	std::cin.get();

	return 0;
}

void ComputeAccelerationsCPU(const std::vector<Particle>& particles, const std::vector<QuadTree>& tree, float* new_accelerations, SimulationSettings settings)
{
	constexpr float G = 6.6743e-9f;
	const float theta_squared = settings.theta * settings.theta;

	#pragma omp parallel for
	for (int i = 0; i < particles.size(); i++)
	{
		const Particle& particle = particles[i];
		glm::vec2 new_acceleration{ 0, 0 };

		std::vector<uint32_t> stack{ 0 };
		stack.reserve(3 * settings.maximum_tree_depth + 1);

		while (!stack.empty())
		{
			uint32_t node_index = stack.back();
			const QuadTree& node = tree[node_index];
			stack.pop_back();

			auto com_displacement = node.GetCenterOfMass() - particle.position;
			auto com_distance_squared = glm::dot(com_displacement, com_displacement);
			auto node_width_squared = node.GetExtents().x * node.GetExtents().x;

			if (node_width_squared < com_distance_squared * theta_squared)
			{
				float inv_r3 = 1.0f / glm::sqrt(com_distance_squared + settings.softening);
				inv_r3 *= inv_r3 * inv_r3 * G * node.GetTotalMass();
				new_acceleration += inv_r3 * com_displacement;
			}
			else if (node.HasChildren())
			{
				for (auto child_index : node.GetChildren())
					stack.push_back(child_index);
			}
			else
			{
				for (auto& other_particle : node.GetParticles())
				{
					auto displacement = other_particle.position - particle.position;
					auto distance_squared = glm::dot(displacement, displacement);
					float inv_r3 = 1.0f / glm::sqrt(distance_squared + settings.softening);
					inv_r3 *= inv_r3 * inv_r3 * G * other_particle.mass;
					new_acceleration += inv_r3 * displacement;
				}
			}
		}

		new_accelerations[2 * i] = new_acceleration.x;
		new_accelerations[2 * i + 1] = new_acceleration.y;
	}
}

float rand_range(float min_inclusive, float max_inclusive)
{
	//static std::mt19937_64 generator{ std::random_device{}() };
	static std::mt19937_64 generator{ 69420 };
	std::uniform_real_distribution<float> dist(min_inclusive, max_inclusive);
	return dist(generator);
}
