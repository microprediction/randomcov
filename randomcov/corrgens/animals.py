import numpy as np
import matplotlib.pyplot as plt


# Correlation of sizes of interacting animals that grow when they meet


class Animal:
    def __init__(self, world_size):
        # Continuous attributes: position, velocity, size, energy
        self.position = np.random.rand(2) * world_size  # Random position in the world
        self.velocity = (np.random.rand(2) - 0.5) * 2  # Random initial velocity
        self.size = np.random.rand() + 0.5  # Size of the animal, random between 0.5 and 1.5
        self.energy = np.random.rand() * 100  # Initial energy

    def move(self, dt=0.1):
        # Random wandering (adding noise to the velocity)
        self.velocity += (np.random.rand(2) - 0.5) * dt
        self.position += self.velocity * dt

    def interact(self, other, interaction_radius):
        # Check if the other animal is within interaction distance
        distance = np.linalg.norm(self.position - other.position)
        if distance < interaction_radius:
            # Example interaction: size increases when close to another animal
            self.size += 0.01  # Increase size slightly
            other.size += 0.01  # Mutual interaction

class World:
    def __init__(self, n_animals):
        self.world_size = int(np.sqrt(10*n_animals))
        self.animals = [Animal(self.world_size) for _ in range(n_animals)]
        self.interaction_radius = 0.5  # How close animals need to be to interact
        self.size_history = np.zeros((n_animals, 0))  # To store size over time

    def update(self, dt=0.1):
        # Update the position and interactions of all animals
        for animal in self.animals:
            animal.move(dt)
        
        # Check for interactions
        for i, animal in enumerate(self.animals):
            for j, other in enumerate(self.animals):
                if i != j:
                    animal.interact(other, self.interaction_radius)

        # Track sizes of animals at the current step
        sizes = np.array([animal.size for animal in self.animals])
        self.size_history = np.hstack((self.size_history, sizes[:, np.newaxis]))

    def simulate(self, num_steps=1000):
        for _ in range(num_steps):
            self.update()

    def compute_correlation_matrix(self):
        # Compute the correlation matrix of size evolution over time
        return np.corrcoef(self.size_history)

def animals_corr(n):
      n_animals = 100
      world = World(n_animals)
      world.simulate(num_steps=500)
      return world.compute_correlation_matrix()

