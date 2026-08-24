# N-Body Gravity Simulator

This is a cross-platform Newtonian gravity N-body simulator. It uses the Barnes-Hut approximation to compute forces in O(N log N) time, rather than the O(N<sup>2</sup>) time required for computing all interaction pairs. Velocity Verlet integration is used for its superior accuracy (compared to naive Euler integration), numerical stability, and conservation properties. CUDA and OpenMP are utilized to significantly improve performance for large particle counts.

### The Barnes-Hut Approximation

For N particles, computing the gravitational force between all particle pairs takes O(N<sup>2</sup>) time, which becomes prohibitively expensive for large N. The Barnes-Hut approximation reduces this complexity to O(N log N) by recursively subdividing space into cells which are arranged in a tree structure. Each leaf cell of this tree stores the particles that inhabit the corresponding region of space, while the internal cells store the total mass and center of mass determined (recursively) from their children. This facilitates treating far-away groups of particles as a single particle, reducing a potentially large number of interactions to a single computation. Whether a cell is sufficiently far away to be approximated depends on a parameter θ. When calculating the force on a particle, the tree is traversed, and a cell is treated as a single particle if s/d < θ, where s is the width of the cell and d is the distance between the particle and the cell's center of mass. Otherwise, if the cell is a leaf cell, all pair interactions with its particles are computed, and if the cell is an internal cell, its children are traversed. For θ=0, the algorithm degenerates into the direct O(N<sup>2</sup>) computation.


# Building
This project requires CMake (>=3.25) and a C++20 compiler that supports OpenMP. It can be built as follows:
```
$ mkdir build
$ cmake -S . -B build
$ cmake --build build --parallel
```
The `--parallel` flag is optional, but speeds the build up. 