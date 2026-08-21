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
#include "configuration.h"
#include "particle.h"
#include "renderer.h"
#include "quad_tree.h"
#include "gpu_info.h"
#include "timer.h"
#include "cuda.cuh"

float zoom = 1.0f;

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

	window = glfwCreateWindow(1280, 720, "Barnes-Hut Gravity Simulator", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	glfwSwapInterval(0);

	glfwSetScrollCallback(window, [](GLFWwindow* window, double scroll_x, double scroll_y) {
		int scroll = static_cast<int>(scroll_y);
		if (scroll > 0)
			zoom *= 1.1f;
		else
			zoom /= 1.1f;

		Renderer::SetZoom(zoom);
	});

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

	SimulationSettings settings {
		.max_steps = 0,
		.leaf_capacity = 64,
		.maximum_tree_depth = 8,
		.threads_per_block = 256,
		.softening = 1e-6f,
		.theta = 0.5f,
		.time_step = 0.01f
	};

	ConfigurationSettings configuration_settings {
		.type = ConfigurationType::UNIFORM_SQUARE,
		.particle_count = 5'000,
		.variance = 0.05f,
		.plummer_radius = 0.2f
	};
	std::vector<Particle> initial_configuration = GenerateUniformSquare(10'000);
	std::vector<Particle> particles = initial_configuration;

	//
	// Simulation
	//

	CUDAContext cuda_context;
	if (gpu_info.cuda_supported)
		cuda_context = CUDAContext(particles.size(), settings.maximum_tree_depth);

	Renderer::Init(particles.size());
	Renderer::SetZoom(1.0f);

	float* new_accelerations = new float[2 * particles.size()];
	float max_position_coord = 1.0f;

	unsigned long long step = 0;
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		if (reset)
		{
			zoom = 1.0f;
			Renderer::SetZoom(zoom);

			if (initial_configuration.size() != particles.size())
				Renderer::SetParticleCount(initial_configuration.size());

			particles = initial_configuration;
			for (int i = 0; i < particles.size(); i++)
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

			Timer::Start("Construction");
			for (int i = 0; i < particles.size(); i++)
			{
				Particle& particle = particles[i];
				particle.position += particle.velocity * settings.time_step + 0.5f * particle.acceleration * settings.time_step * settings.time_step;
				Renderer::SetParticlePosition(i, particle.position);

				max_position_coord = glm::max(max_position_coord, glm::abs(particle.position.x));
				max_position_coord = glm::max(max_position_coord, glm::abs(particle.position.y));
			}
			QuadTree::GetPool().emplace_back(glm::dvec2{ 0, 0 }, 2.1f * max_position_coord);
			QuadTree::GetPool()[0].Build(std::span<Particle>(particles.begin(), particles.end()));

			Timer::Start("Mass");
			QuadTree::GetPool()[0].CalculateMass();
			Timer::End();
			Timer::End();

			//
			// Compute accelerations
			//

			Timer::Start("Acceleration");
			if (gpu_info.cuda_supported)
				ComputeAccelerationsCUDA(cuda_context, particles, QuadTree::GetPool(), new_accelerations, settings); // Blocks while GPU finishes
			else
				ComputeAccelerationsCPU(particles, QuadTree::GetPool(), new_accelerations, settings);
			Timer::End();

			//
			// Integrate (velocity Verlet)
			//

			Timer::Start("Integration");
			#pragma omp parallel for
			for (int i = 0; i < particles.size(); i++)
			{
				glm::vec2 new_acceleration{ new_accelerations[2 * i], new_accelerations[2 * i + 1] };
				Particle& particle = particles[i];
				particle.velocity += 0.5f * (particle.acceleration + new_acceleration) * settings.time_step;
				particle.acceleration = new_acceleration;
			}
			Timer::End();

			step++;
			if (settings.max_steps > 0 && step >= settings.max_steps)
				running = false;
		}

		//
		// UI/Visualization
		//

		int control_width = static_cast<int>(0.25f * io.DisplaySize.x);
		int view_width = static_cast<int>(io.DisplaySize.x) - control_width;

		ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
		ImGui::SetNextWindowSize(ImVec2(control_width, io.DisplaySize.y), ImGuiCond_Once);
		ImGui::SetNextWindowSizeConstraints(ImVec2(0, io.DisplaySize.y), ImVec2(io.DisplaySize.x, io.DisplaySize.y));

		ImGui::Begin("Controls", 0, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove);

		ImGui::Text("Simulation Parameters");

		ImGui::BeginDisabled(running);

		ImGui::InputScalar("Max Steps", ImGuiDataType_U32, &settings.max_steps);
		ImGui::InputScalar("Leaf Capacity", ImGuiDataType_U32, &settings.leaf_capacity, &leaf_capacity_step, &leaf_capacity_step);
		ImGui::InputScalar("Maximum Tree Depth", ImGuiDataType_U32, &settings.maximum_tree_depth, &maximum_tree_depth_step, &maximum_tree_depth_step);
		if (gpu_info.cuda_supported)
		{
			ImGui::InputScalar("Threads/Block", ImGuiDataType_U32, &settings.threads_per_block, &threads_per_block_step, &threads_per_block_step);
			settings.threads_per_block = glm::min(settings.threads_per_block, gpu_info.max_threads_per_block);
		}
		ImGui::InputFloat("Softening", &settings.softening, 0, 0, "%.6f");
		ImGui::InputFloat("Theta", &settings.theta, 0.0f, 2.0f, "%.2f");
		ImGui::InputFloat("Time Step", &settings.time_step, 0, 0, "%.6f");

		ImGui::EndDisabled();

		ImGui::Text("Initial Configuration");

		ImGui::BeginDisabled(running);

		// Configuration type
		auto current_config = ConfigTypeToName(configuration_settings.type);
		if (ImGui::BeginCombo("##combo", current_config))
		{
			for (auto type : GetConfigTypes())
			{
				auto config_name = ConfigTypeToName(type);
				bool already_selected = (current_config == config_name);

				if (ImGui::Selectable(config_name, already_selected))
				{
					current_config = config_name;
					if (!already_selected) // Just changed configuration
					{
						configuration_settings.type = ConfigNameToType(config_name);

						if (configuration_settings.type == ConfigurationType::UNIFORM_SQUARE)
							initial_configuration = GenerateUniformSquare(configuration_settings.particle_count);
						else if (configuration_settings.type == ConfigurationType::UNIFORM_DISK)
							initial_configuration = GenerateUniformDisk(configuration_settings.particle_count);
						else if (configuration_settings.type == ConfigurationType::GAUSSIAN_CLOUD)
							initial_configuration = GenerateGaussianCloud(configuration_settings.particle_count, configuration_settings.variance);
						else if (configuration_settings.type == ConfigurationType::PLUMMER_DISK)
							initial_configuration = GeneratePlummerDisk(configuration_settings.particle_count, configuration_settings.plummer_radius);

						reset = true;
					}
				}

				if (already_selected)
					ImGui::SetItemDefaultFocus();
			}
			ImGui::EndCombo();
		}

		// Configuration parameters
		if (configuration_settings.type == ConfigurationType::UNIFORM_SQUARE)
		{
			if (ImGui::InputInt("# Particles", &configuration_settings.particle_count))
			{
				initial_configuration = GenerateUniformSquare(configuration_settings.particle_count);
				reset = true;
			}
		}
		else if (configuration_settings.type == ConfigurationType::UNIFORM_DISK)
		{
			if (ImGui::InputInt("# Particles", &configuration_settings.particle_count))
			{
				initial_configuration = GenerateUniformDisk(configuration_settings.particle_count);
				reset = true;
			}
		}
		else if (configuration_settings.type == ConfigurationType::GAUSSIAN_CLOUD)
		{
			if (ImGui::InputInt("# Particles", &configuration_settings.particle_count) || ImGui::InputFloat("Variance", &configuration_settings.variance))
			{
				initial_configuration = GenerateGaussianCloud(configuration_settings.particle_count, configuration_settings.variance);
				reset = true;
			}
		}
		else if (configuration_settings.type == ConfigurationType::PLUMMER_DISK)
		{
			if (ImGui::InputInt("# Particles", &configuration_settings.particle_count) || ImGui::InputFloat("Plummer Radius", &configuration_settings.plummer_radius))
			{
				initial_configuration = GeneratePlummerDisk(configuration_settings.particle_count, configuration_settings.plummer_radius);
				reset = true;
			}
		}

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

		ImGui::Begin("Visualization", 0, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

		auto avail = ImGui::GetContentRegionAvail();
		float size = glm::max(avail.x, avail.y);

		if (running)
			Timer::Start("Rendering");

		Renderer::Resize(avail.x, avail.y);
		Renderer::Render();

		if (running)
			Timer::End();

		ImGui::Image(
			(ImTextureID)Renderer::GetTexture(),
			ImVec2(size, size)
		);

		ImGui::End();

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(window);
	}

	auto construction_time = Timer::GetElapsed<std::chrono::milliseconds>("Construction");
	auto mass_time = Timer::GetElapsed<std::chrono::milliseconds>("Mass");
	auto integration_time = Timer::GetElapsed<std::chrono::milliseconds>("Integration");
	auto acceleration_time = Timer::GetElapsed<std::chrono::milliseconds>("Acceleration");
	auto render_time = Timer::GetElapsed<std::chrono::milliseconds>("Rendering");

	double total_time = static_cast<double>(construction_time + integration_time + acceleration_time + render_time);
	std::cout << "Computed " << step << " steps for " << particles.size() << " particles in " << (total_time / 1'000.0) << "s" << std::endl;
	std::cout << "Construction: " << construction_time << "ms (" << (100.0 * static_cast<double>(construction_time) / total_time) << "%)" << std::endl;
	std::cout << "    Mass Calculation: " << mass_time << "ms (" << (100.0 * static_cast<double>(mass_time) / static_cast<double>(construction_time)) << "% of construction)" << std::endl;
	std::cout << "Acceleration Computation: " << acceleration_time << "ms (" << (100.0 * static_cast<double>(acceleration_time) / total_time) << "%)" << std::endl;
	std::cout << "Integration: " << integration_time << "ms (" << (100.0 * static_cast<double>(integration_time) / total_time) << "%)" << std::endl;
	std::cout << "Rendering: " << render_time << "ms (" << (100.0 * static_cast<double>(render_time) / total_time) << "%)" << std::endl;

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
			auto node_width_squared = node.GetWidth() * node.GetWidth();

			if (node_width_squared < com_distance_squared * theta_squared)
			{
				float inv_r3 = 1.0f / glm::sqrt(com_distance_squared + settings.softening);
				inv_r3 *= inv_r3 * inv_r3 * G * node.GetTotalMass();
				new_acceleration += inv_r3 * com_displacement;
			}
			else if (node.HasChildren())
			{
				stack.push_back(node.GetFirstChildIndex());
				stack.push_back(node.GetFirstChildIndex() + 1);
				stack.push_back(node.GetFirstChildIndex() + 2);
				stack.push_back(node.GetFirstChildIndex() + 3);
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
