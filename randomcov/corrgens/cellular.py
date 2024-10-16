import numpy as np

# Not tested yet. 

def cellular_corr(n: int, grid_size: int = 10, num_steps: int = None, plant_growth_prob: float = 0.02,
                  herbivore_reproduction_energy: int = 10, carnivore_reproduction_energy: int = 20) -> np.ndarray:
    """
    Generate a correlation matrix using a cellular automata-inspired ecosystem model on a toroidal grid.
    
    Parameters:
    -----------
    n : int
        The number of agents (grid cells) to track for correlation.
    grid_size : int, optional
        Size of the grid (default is 10x10).
    num_steps : int, optional
        The number of steps to simulate (default is 100).
    plant_growth_prob : float, optional
        Probability of plant cells growing (default is 0.02).
    herbivore_reproduction_energy : int, optional
        The energy required for herbivores to reproduce (default is 10).
    carnivore_reproduction_energy : int, optional
        The energy required for carnivores to reproduce (default is 20).
    
    Returns:
    --------
    corr_matrix : np.ndarray
        The resulting correlation matrix of cell states after simulation.
    """
    EMPTY, PLANT, HERBIVORE, CARNIVORE = 0, 1, 2, 3
                    
    if num_steps is None:
      num_steps = 2*n 
    
    # Initialize the grid and energy levels for cells
    grid = np.zeros((grid_size, grid_size), dtype=int)
    energy = np.zeros((grid_size, grid_size), dtype=int)

    # Randomly seed plants, herbivores, and carnivores
    for i in range(grid_size):
        for j in range(grid_size):
            if np.random.random() < 0.05:
                grid[i, j] = PLANT
            elif np.random.random() < 0.02:
                grid[i, j] = HERBIVORE
                energy[i, j] = herbivore_reproduction_energy // 2
            elif np.random.random() < 0.01:
                grid[i, j] = CARNIVORE
                energy[i, j] = carnivore_reproduction_energy // 2

    # Function to update the grid based on local rules
    def update(grid, energy):
        new_grid = np.copy(grid)
        new_energy = np.copy(energy)
        
        for i in range(grid_size):
            for j in range(grid_size):
                if grid[i, j] == PLANT:
                    if np.random.random() < plant_growth_prob:
                        ni, nj = (i + np.random.choice([-1, 1])) % grid_size, (j + np.random.choice([-1, 1])) % grid_size
                        if new_grid[ni, nj] == EMPTY:
                            new_grid[ni, nj] = PLANT
                elif grid[i, j] == HERBIVORE:
                    found_food = False
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = (i + di) % grid_size, (j + dj) % grid_size  # Wrap around on the torus
                        if grid[ni, nj] == PLANT:
                            new_grid[ni, nj] = HERBIVORE
                            new_grid[i, j] = EMPTY
                            found_food = True
                            new_energy[ni, nj] += 5  # Gain energy from eating
                            break
                    if not found_food:
                        new_energy[i, j] -= 1
                        if new_energy[i, j] <= 0:
                            new_grid[i, j] = EMPTY
                elif grid[i, j] == CARNIVORE:
                    found_prey = False
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = (i + di) % grid_size, (j + dj) % grid_size  # Wrap around on the torus
                        if grid[ni, nj] == HERBIVORE:
                            new_grid[ni, nj] = CARNIVORE
                            new_grid[i, j] = EMPTY
                            found_prey = True
                            new_energy[ni, nj] += 10  # Gain energy from eating
                            break
                    if not found_prey:
                        new_energy[i, j] -= 1
                        if new_energy[i, j] <= 0:
                            new_grid[i, j] = EMPTY
        return new_grid, new_energy

    # Track the states over time
    states = np.zeros((num_steps, grid_size, grid_size))
    for step in range(num_steps):
        grid, energy = update(grid, energy)
        states[step] = grid.flatten()

    # Select n random cells from the grid to calculate their state correlations
    selected_cells = np.random.choice(grid_size * grid_size, size=n, replace=False)
    selected_states = states[:, selected_cells]

    # Compute the correlation matrix for the selected cells
    corr_matrix = np.corrcoef(selected_states.T)

    return corr_matrix

# Example usage:
n = 10  # Number of cells to track
correlation_matrix = cellular_corr(n, grid_size=20, num_steps=200)
print("Correlation Matrix:\n", correlation_matrix)
