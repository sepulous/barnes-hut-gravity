
# N-Body Gravity Simulator

<div align="center">

![Program](https://i.ibb.co/vxL1Tfvs/program.png)
</div>

This is a cross-platform Newtonian gravity N-body simulator. It uses the Barnes-Hut approximation to compute forces in O(N log N) time, rather than the O(N<sup>2</sup>) time required for computing all interaction pairs. Velocity Verlet integration is used for its superior accuracy (compared to Euler integration), numerical stability, and conservation properties. CUDA and OpenMP are utilized to significantly improve performance for large particle counts.

### The Barnes-Hut Approximation

For N particles, computing the gravitational force between all particle pairs takes O(N<sup>2</sup>) time, which becomes prohibitively expensive for large N. The Barnes-Hut approximation reduces this complexity to O(N log N) by recursively subdividing space into cells which are arranged in a tree structure. Each leaf cell of this tree stores the particles that inhabit the corresponding region of space, while the internal cells store the total mass and center of mass determined (recursively) from their children. This facilitates treating far-away groups of particles as a single particle, reducing a potentially large number of interactions to a single computation. Whether a cell is sufficiently far away to be approximated depends on a parameter θ. When calculating the force on a particle, the tree is traversed, and a cell is treated as a single particle if s/d < θ, where s is the width of the cell and d is the distance between the particle and the cell's center of mass. Otherwise, if the cell is a leaf cell, all pair interactions with its particles are computed, and if the cell is an internal cell, its children are traversed. For θ=0, the algorithm degenerates into the direct O(N<sup>2</sup>) computation.

Clearly, θ controls the tradeoff between accuracy and efficiency. There are diminishing returns in runtime efficiency for higher θ values, which tend to taper off around θ=0.6:

<div align="center">

![Theta Tradeoff](https://i.ibb.co/kV02qYM1/theta-tradeoff.png)

</div>

The following illustrates that the overhead of constructing and using the tree is significantly outweighed by the reduction in work that it allows:
<div align="center">
<img width="60%" src="https://i.ibb.co/C3BVBwsV/runtime-vs-particle-count.png" alt="Runtime vs Particle Count" />
</div>

### Velocity Verlet Integration

<a href="https://en.wikipedia.org/wiki/Euler_method" target="_blank">Euler integration</a> has a number of properties that make it unsuitable for physical simulations. First, it has a global error in position and velocity on the order of O(Δt), which means that it generally requires a very small time step for acceptable accuracy, thus requiring a longer runtime compared to higher-order methods. Additionally, Euler integration can exhibit numeric instability in certain situations, including orbital motion under Newtonian gravity. Lastly, Euler integration fails to conserve a system's energy, leading to qualitatively incorrect behavior due to energy drift over time. While these last two problems can be remedied through a <a href="https://en.wikipedia.org/wiki/Semi-implicit_Euler_method" target="_blank">modification</a>, the resulting global error is still first-order.

This project uses <a href="https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet" target="_blank">velocity Verlet integration</a>. Velocity Verlet is as simple to compute as Euler, but has a lower global error of O(Δt<sup>2</sup>) and better numerical stability. It also conserves a modified form of the system's energy, thus preserving the expected qualitative behavior. Using a higher-order method such as this allows one to achieve better accuracy for the same runtime (by using the same time step), or faster runtime with the same accuracy (by increasing the time step). The following illustrates the difference in energy conservation between Euler and velocity Verlet:

<div align="center">
<img width="60%" src="https://i.ibb.co/DP8MFWgY/energy-vs-steps.png" alt="Energy vs Steps" />
</div>

# Building
This project requires CMake (>=3.25) and a C++20 compiler that supports OpenMP. It can be built as follows:
```
$ mkdir build
$ cmake -S . -B build
$ cmake --build build --parallel
```
The `--parallel` flag is optional, but speeds the build up. 
