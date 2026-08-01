#include <iostream>
#include <vector>
#include <random>
#include <thread>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "helpers.h"
#include "shader.h"
#include "quad_tree.h"

using namespace std::chrono_literals;

#define RENDER 1
#define PARALLEL 1
#define MAX_STEPS 1000

double rand_range(double min_inclusive, double max_inclusive);
Vec2 GetAccelerationAtPoint(Point* point, QuadTree& tree);

constexpr size_t NUM_POINTS = 10'000;
constexpr double TIME_STEP = 0.01;
constexpr double TIME_STEP_SQUARED = TIME_STEP * TIME_STEP;
constexpr double THETA = 0.8; // width/distance threshold for quad tree cells
constexpr double SOFTENING = 1e-3;
constexpr double METERS_PER_UNIT = 1.0;
constexpr double G = 100.0 * 6.67408e-11 / (METERS_PER_UNIT * METERS_PER_UNIT);

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

	window = glfwCreateWindow(800, 800, "Barnes-Hut Simulation", NULL, NULL);
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
		NUM_POINTS * (2 * sizeof(float)),
		nullptr,
		GL_MAP_WRITE_BIT |
		GL_MAP_PERSISTENT_BIT |
		GL_MAP_COHERENT_BIT
	);

	float* gpu_positions = static_cast<float*>(
		glMapNamedBufferRange(
			particle_vbo,
			0,
			NUM_POINTS * (2 * sizeof(float)),
			GL_MAP_WRITE_BIT |
			GL_MAP_PERSISTENT_BIT |
			GL_MAP_COHERENT_BIT
		)
	);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
	glEnableVertexAttribArray(0);

	Shader shader_program(
		"C:\\Users\\miner\\source\\repos\\BarnesHut\\src\\point_frag.glsl",
		"C:\\Users\\miner\\source\\repos\\BarnesHut\\src\\point_vert.glsl",
		{
			{2, GL_FLOAT}
		}
	);
	shader_program.Compile();

	//
	// Initial configuration
	//
	std::vector<Point> points;
	points.reserve(NUM_POINTS);
	for (int i = 0; i < NUM_POINTS; i++)
	{
		points.push_back({
			.position = {
				.x = rand_range(-1.0, 1.0),
				.y = rand_range(-1.0, 1.0)
			},
			.velocity = {0.0, 0.0},
			.acceleration = {0.0, 0.0},
			.mass = rand_range(1.0, 50.0)
		});
		points[i].next_position = points[i].position;
	}
	//for (int i = 0; i < NUM_POINTS; i++)
	//{
	//	points.push_back({
	//		.position = {
	//			.x = 0.99 * std::cos(i * (2.0 * 3.141593 / NUM_POINTS)),
	//			.y = 0.99 * std::sin(i * (2.0 * 3.141593 / NUM_POINTS))
	//		},
	//		.velocity = {0.0, 0.0},
	//		.acceleration = {0.0, 0.0},
	//		.mass = 5.0
	//	});
	//	points[i].next_position = points[i].position;
	//}

	double initial_total_energy = 0;
	for (int i = 0; i < NUM_POINTS; i++)
	{
		Point& p1 = points[i];

		// Kinetic
		initial_total_energy += 0.5 * p1.mass * p1.velocity.dot(p1.velocity);

		for (int j = i + 1; j < NUM_POINTS; j++)
		{
			Point& p2 = points[j];

			double r = (p1.position - p2.position).length();
			double soft_distance = 1.0 / std::sqrt(r * r + SOFTENING * SOFTENING);

			// Potential
			initial_total_energy += (G * p1.mass * p2.mass) / soft_distance;
		}
	}
	printf("Initial total energy: %f\n", initial_total_energy);

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

		//
		// Update positions and construct quad tree
		//
		QuadTree tree({ 0, 0 }, { 1, 1 }); // centered at (0, 0), goes from (-1, -1) to (1, 1)
		for (int i = 0, j = 0; i < NUM_POINTS; i++, j += 2)
		{
			Point& point = points[i];
			point.position = point.next_position;
			#if RENDER
			gpu_positions[j] = static_cast<float>(point.position.x);
			gpu_positions[j + 1] = static_cast<float>(point.position.y);
			#endif
			tree.Add(&point);
		}
		tree.CalculateMass();

		//
		// Integrate
		//
		#if PARALLEL
		#pragma omp parallel for
		for (int i = 0; i < NUM_POINTS; i++)
		{
			Point& point = points[i];
		#else
		for (auto& point : points)
		{
		#endif
			Vec2 new_acceleration = GetAccelerationAtPoint(&point, tree);

			point.next_position = {
				.x = point.position.x + point.velocity.x * TIME_STEP + 0.5 * point.acceleration.x * TIME_STEP_SQUARED,
				.y = point.position.y + point.velocity.y * TIME_STEP + 0.5 * point.acceleration.y * TIME_STEP_SQUARED
			};
			point.velocity = {
				.x = point.velocity.x + 0.5 * (point.acceleration.x + new_acceleration.x) * TIME_STEP,
				.y = point.velocity.y + 0.5 * (point.acceleration.y + new_acceleration.y) * TIME_STEP
			};
			point.acceleration = new_acceleration;
		}

		#if MAX_STEPS
		step++;
		printf("Step: %i\n", step);
		if (step == MAX_STEPS)
		{
			printf("Computed %i steps with %i particles in %f seconds\n", MAX_STEPS, NUM_POINTS, glfwGetTime() - start_time);
			break;
		}
		#endif

		//
		// Render
		//
		#if RENDER
		glClear(GL_COLOR_BUFFER_BIT);

		shader_program.Use();
		glBindVertexArray(particle_vao);
		glDrawArrays(GL_POINTS, 0, NUM_POINTS);

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		glfwSwapBuffers(window);
		#endif
	}

	double final_total_energy = 0;
	double angular_momentum = 0;
	for (int i = 0; i < NUM_POINTS; i++)
	{
		Point& p1 = points[i];
		angular_momentum += p1.mass * (p1.position.x * p1.velocity.y - p1.position.y * p1.velocity.x);

		// Kinetic
		final_total_energy += 0.5 * p1.mass * p1.velocity.dot(p1.velocity);

		for (int j = i + 1; j < NUM_POINTS; j++)
		{
			Point& p2 = points[j];

			double r = (p1.position - p2.position).length();
			double soft_distance = 1.0 / std::sqrt(r * r + SOFTENING * SOFTENING);

			// Potential
			final_total_energy += (G * p1.mass * p2.mass) / soft_distance;
		}
	}
	printf("Final total energy: %f\n", final_total_energy);
	printf("Angular momentum: %f\n", angular_momentum);

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

Vec2 GetAccelerationAtPoint(Point* point, QuadTree& tree)
{
	Vec2 acceleration = { 0, 0 };

	Vec2 com_displacement = tree.GetCenterOfMass() - point->position;
	double com_distance = com_displacement.length();
	double width = 2.0 * tree.GetExtents().x;
	if ((tree.GetPoint() && tree.GetPoint() != point) || width / com_distance < THETA) // Use whole node in calculation (leaf node or sufficiently far)
	{
		double soft_inv_distance = com_distance * com_distance + SOFTENING * SOFTENING;
		soft_inv_distance *= soft_inv_distance * soft_inv_distance;
		soft_inv_distance = 1.0 / std::sqrt(soft_inv_distance);
		acceleration += (G * tree.GetTotalMass() * soft_inv_distance) * com_displacement;
	}
	else if (tree.HasChildren()) // Internal node; look at children
	{
		for (auto& child : tree.GetChildren())
			acceleration += GetAccelerationAtPoint(point, child);
	}

	return acceleration;
}
