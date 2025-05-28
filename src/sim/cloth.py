import taichi as ti
import numpy as np

# https://docs.taichi-lang.org/
# https://ambientcg.com/


@ti.data_oriented
class ClothSimulation:
    def __init__(self, n: int, time: float):
        self.n = n  # Define the grid size (n x n)
        self.time = time

        # Define the grid spacing and simulation parameters
        self.quad_size = 1.0 / self.n  # Distance between adjacent mass points
        self.dt: ti.types = 4e-2 / self.n  # Time step size
        self.substeps = int(1 / 60 // self.dt)  # Number of substeps per frame

        self.gravity = ti.Vector(
            [0, -9.8, 0]
        )  # Define gravity acting in the negative y direction

        # Define material properties
        self.spring_Y = 8e4  # Elastic coefficient of springs
        self.dashpot_damping = 3e4  # Damping coefficient for velocity difference
        self.drag_damping = 1  # Air resistance damping

        # Define Taichi fields for position and velocity of mass points
        self.cloth_position = ti.Vector.field(
            3, dtype=float, shape=(self.n, self.n)
        )  # Position field
        self.cloth_velocity = ti.Vector.field(
            3, dtype=float, shape=(self.n, self.n)
        )  # Velocity field

        self.num_triangles = (
            (self.n - 1) * (self.n - 1) * 2
        )  # Total number of triangles in the mesh
        self.indices = ti.field(
            int, shape=self.num_triangles * 3
        )  # Mesh connectivity information
        self.vertices = ti.Vector.field(
            3, dtype=float, shape=self.n * self.n
        )  # Vertex positions

        self.fold = ti.field(int, shape=())
        self.fold[None] = False

        self.bending_springs = False  # Enable bending springs
        self.winkle_mask = ti.field(int, shape=(self.n - 1, self.n - 1))

        self.current_t = 0.0 # Used to keep trace of simulation time (sec)
        self.initialize_mesh_indices()  # Initialize the mesh indices
        self.initialize_mass_points()  # Initialize mass points
        self.initialize_winkle_mask()


        # Define spring connections between mass points
        self.spring_offsets = []
        if self.bending_springs:
            # Iterate over a 3x3 grid from -1 to 1
            for i in range(-1, 1 + 1):
                for j in range(-1, 1 + 1):
                    # add the spring offset expect for the center

                    # o  o  o
                    #  \ | /
                    # o--o--o
                    #  / | \
                    # o  o  o
                    if (i, j) != (0, 0):
                        self.spring_offsets.append(ti.Vector([i, j]))
        else:
            # Iterate over a 5x5 grid from -2 to 2
            for i in range(-2, 2 + 1):
                for j in range(-2, 2 + 1):
                    # add the spring offset except:
                    # 1. for the center
                    # 2. and only in a cross like pattern

                    #       o
                    #       |
                    #    o  o  o
                    #     \ | /
                    # o--o--o--o--o
                    #     / | \
                    #    o  o  o
                    #       |
                    #       o
                    if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
                        self.spring_offsets.append(ti.Vector([i, j]))

    @ti.kernel
    def initialize_mass_points(self):
        """Initialize the mass points with random offsets and zero initial velocity."""
        for i, j in self.cloth_position:
            self.cloth_position[i, j] = [
                # x-position
                i * self.quad_size - (self.n * self.quad_size) / 2,
                # y-position (cloth starts above the origin)
                2.0 + (ti.random() - 0.5) * 0.0032,
                # z-position
                j * self.quad_size
                - (self.n * self.quad_size) / 2,
            ]
            self.cloth_velocity[i, j] = [0, 0, 0]  # Initialize velocity to zero

    @ti.kernel
    def initialize_mesh_indices(self):
        """Initialize mesh indices for rendering the cloth surface."""
        for i, j in ti.ndrange(self.n - 1, self.n - 1):
            quad_id = (i * (self.n - 1)) + j  # Unique ID for each quad
            # We add two vertex so we need enough memory indexses

            # First triangle of the square
            self.indices[quad_id * 6 + 0] = i * self.n + j
            self.indices[quad_id * 6 + 1] = (i + 1) * self.n + j
            self.indices[quad_id * 6 + 2] = i * self.n + (j + 1)

            # Second triangle of the square
            self.indices[quad_id * 6 + 3] = (i + 1) * self.n + j + 1
            self.indices[quad_id * 6 + 4] = i * self.n + (j + 1)
            self.indices[quad_id * 6 + 5] = (i + 1) * self.n + j

    @ti.kernel
    def initialize_winkle_mask(self):
        for i, j in ti.ndrange(self.n - 1, self.n - 1):
            self.winkle_mask[i, j] = 0

    @ti.func
    def gravity_func(self):
        # Apply gravity
        for i in ti.grouped(self.cloth_position):
            # velocity = velocity + acceleration * dt
            self.cloth_velocity[i] += self.gravity * self.dt

    @ti.func
    def internal_forces(self):
        # Initial force exerted to a specific mass point
        # AKA compute internal spring forces
        for i in ti.grouped(self.cloth_position):
            force = ti.Vector([0.0, 0.0, 0.0])

            # Traverse the surrounding mass points'
            # AKA iterate over neighboring mass points
            for spring_offset in ti.static(self.spring_offsets):
                # j is the *absolute* index of an 'influential' point
                # Note that j is a 2-dimensional vector here
                # AKA a neighboring mass point
                j = i + spring_offset

                # If the 'influential` point is in the n x n grid,
                # then work out the internal force that it exerts
                # on the current mass point
                # AKA we are checking if the offset is within the NxN grid both for x and y
                if 0 <= j[0] < self.n and 0 <= j[1] < self.n:
                    # The relative displacement of the two points
                    # The internal force is related to it
                    displacement_pos_ij = (
                        self.cloth_position[i] - self.cloth_position[j]
                    )

                    # The relative movement of the two points
                    displacement_vec_ij = (
                        self.cloth_velocity[i] - self.cloth_velocity[j]
                    )

                    # d is a normalized vector (its norm is 1)
                    # AKA unit vector pointing to the current position displacement
                    d = displacement_pos_ij.normalized()

                    # The "real" distance between the current mass point and its displacement
                    current_dist = displacement_pos_ij.norm()

                    # The original distance
                    original_dist = self.quad_size * float(i - j).norm()

                    # Internal force of the spring
                    # Uses Hooke's Law for Springs
                    # F_sptring = -spring_stiffness * (change of length i procent)
                    force += -self.spring_Y * d * (current_dist / original_dist - 1)

                    # Damping force
                    # 1. displacement_vec_ij.dot(d):
                    # Calculates the velocity component along the spring direction (how fast the points are moving toward or away from each other).

                    # 2. * d:
                    # Scales the damping force to apply it in the direction of the spring (so it affects only the movement along the spring's line).

                    # 3: * dashpot_damping:
                    # Scales the force by the damping coefficient to control how much damping to apply.

                    # 4: * quad_size:
                    # Ensures the force is applied at the correct scale depending on the size of the simulation.
                    force += (
                        -displacement_vec_ij.dot(d)
                        * d
                        * self.dashpot_damping
                        * self.quad_size
                    )

            # Continues to add the velocity caused by the internal forces
            # to the current velocity
            self.cloth_velocity[i] += force * self.dt

    @ti.func
    def simple_air_drag(self):
        for i in ti.grouped(self.cloth_position):
            # Apply air resistance
            self.cloth_velocity[i] *= ti.exp(-self.drag_damping * self.dt)

    @ti.func
    def ground_collision(self):
        for i in ti.grouped(self.cloth_position):
            if self.cloth_position[i].y <= 0:  # Check collision
                normal = ti.Vector([0.0, 1.0, 0.0])  # Normal of the ground
                self.cloth_velocity[i] -= (
                    min(self.cloth_velocity[i].dot(normal), 0) * normal
                )  # Reflect velocity

    @ti.kernel
    def substep(self):
        """Perform one simulation substep: apply forces and update positions."""
        self.gravity_func()
        self.internal_forces()
        self.simple_air_drag()
        self.ground_collision()

        # Update positions based on simulation precision
        for i in ti.grouped(self.cloth_position):
            self.cloth_position[i] += self.dt * self.cloth_velocity[i]

    @ti.kernel
    def update_vertices(self):
        """Update vertex positions for rendering."""
        for i, j in ti.ndrange(self.n, self.n):
            self.vertices[i * self.n + j] = self.cloth_position[i, j]

    @ti.kernel
    def ensure_fold(self):
        for i in ti.grouped(self.cloth_position):
            fold = 0
            if self.cloth_position[i].y >= 0.0037:
                fold = 1
            self.winkle_mask[i] = fold

    def winkle_white_ratio(self):
        mask_np = self.winkle_mask.to_numpy()
        white_pixels = np.sum(mask_np > 0)
        total_pixels = mask_np.size
        ratio = white_pixels / total_pixels
        return ratio

    def simulate(self):
        while self.current_t < self.time:
            for i in range(self.substeps):
                self.substep()
                self.current_t += self.dt

            self.update_vertices()
        
        self.ensure_fold()
        if self.winkle_white_ratio() < 0.03:
            self.reset_simulation()
            self.simulate()
           
    def reset_simulation(self):
        self.fold[None] = False
        self.current_t = 0.0
        self.initialize_mass_points()
        self.initialize_mesh_indices()
        self.initialize_winkle_mask()


def main():
    ti.init(arch=ti.vulkan)
    
    n = 256
    simulation_time = 0.9
    sim = ClothSimulation(n=n, time=simulation_time)

    window = ti.ui.Window("Cloth Simulation", (1024, 1024), vsync=True)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()

    camera.position(0, 1.8, 0)
    camera.lookat(0, 0, 0)
    camera.up(0, 0, 1)

    while sim.current_t < simulation_time:
        for _ in range(sim.substeps):
            sim.substep()
            sim.current_t += sim.dt
        
        sim.update_vertices()
        sim.ensure_fold()

        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(2, 3, 2), color=(1, 1, 1))

        scene.mesh(sim.vertices, indices=sim.indices, color=(0.3, 0.5, 0.8), two_sided=True)

        canvas.set_background_color((0.1, 0.2, 0.3))
        canvas.scene(scene)

        window.show()

if __name__ == "__main__":
    main()