import time
import random
import matplotlib.pyplot as plt
import numpy as np

# Cell class to represent each cell in the maze
class Cell:
    def __init__(self):
        self.walls = {'N': True, 'S': True, 'E': True, 'W': True}
        self.visited = False
        self.entry_exit = False

    def remove_walls(self, next_i, next_j):
        if next_i == -1 and next_j == -1:
            self.walls['N'] = False
        elif next_i == 1 and next_j == 1:
            self.walls['S'] = False
        elif next_i == 0 and next_j == 1:
            self.walls['E'] = False
        elif next_i == 1 and next_j == 0:
            self.walls['W'] = False

    def set_as_entry_exit(self, type, num_rows, num_cols):
        self.entry_exit = type
        if type == "entry":
            self.walls['N'] = False
        elif type == "exit":
            self.walls['S'] = False

    def is_walls_between(self, other):
        return all(self.walls[k] == other.walls[k] for k in self.walls)

# Maze class to create and manipulate the maze
class Maze:
    def __init__(self, num_rows, num_cols):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.grid_size = num_rows * num_cols
        self.grid = [[Cell() for _ in range(num_cols)] for _ in range(num_rows)]
        self.entry_coor = (0, 0)
        self.exit_coor = (num_rows-1, num_cols-1)
        self.generation_path = []

    def find_neighbours(self, row, col):
        neighbours = []
        if row > 0: neighbours.append((row-1, col))
        if row < self.num_rows-1: neighbours.append((row+1, col))
        if col > 0: neighbours.append((row, col-1))
        if col < self.num_cols-1: neighbours.append((row, col+1))
        return neighbours

    def _validate_neighbours_generate(self, neighbours):
        valid_neighbours = []
        for (i, j) in neighbours:
            if not self.grid[i][j].visited:
                valid_neighbours.append((i, j))
        return valid_neighbours

def depth_first_recursive_backtracker(maze, start_coor):
    k_curr, l_curr = start_coor  # Where to start generating
    path = [(k_curr, l_curr)]  # To track path of solution
    maze.grid[k_curr][l_curr].visited = True  # Set initial cell to visited
    visit_counter = 1  # To count number of visited cells
    visited_cells = list()  # Stack of visited cells for backtracking

    print("\nGenerating the maze with depth-first search...")
    time_start = time.time()

    while visit_counter < maze.grid_size:  # While there are unvisited cells
        neighbour_indices = maze.find_neighbours(k_curr, l_curr)  # Find neighbour indices
        neighbour_indices = maze._validate_neighbours_generate(neighbour_indices)

        if neighbour_indices:  # If there are unvisited neighbour cells
            visited_cells.append((k_curr, l_curr))  # Add current cell to stack
            k_next, l_next = random.choice(neighbour_indices)  # Choose random neighbour
            maze.grid[k_curr][l_curr].remove_walls(k_next, l_next)  # Remove walls between neighbours
            maze.grid[k_next][l_next].remove_walls(k_curr, l_curr)  # Remove walls between neighbours
            maze.grid[k_next][l_next].visited = True  # Move to that neighbour
            k_curr = k_next
            l_curr = l_next
            path.append((k_curr, l_curr))  # Add coordinates to part of generation path
            visit_counter += 1

        elif visited_cells:  # If there are no unvisited neighbour cells
            k_curr, l_curr = visited_cells.pop()  # Pop previous visited cell (backtracking)
            path.append((k_curr, l_curr))  # Add coordinates to part of generation path

    print("Number of moves performed: {}".format(len(path)))
    print("Execution time for algorithm: {:.4f}".format(time.time() - time_start))

    maze.grid[maze.entry_coor[0]][maze.entry_coor[1]].set_as_entry_exit("entry", maze.num_rows-1, maze.num_cols-1)
    maze.grid[maze.exit_coor[0]][maze.exit_coor[1]].set_as_entry_exit("exit", maze.num_rows-1, maze.num_cols-1)

    for i in range(maze.num_rows):
        for j in range(maze.num_cols):
            maze.grid[i][j].visited = False  # Set all cells to unvisited before returning grid

    maze.generation_path = path

def binary_tree(maze, start_coor):
    # store the current time
    time_start = time.time()

    # repeat the following for all rows
    for i in range(maze.num_rows):
        # check if we are in top row
        if i == maze.num_rows - 1:
            # remove the right wall for this, because we can't remove the top wall
            for j in range(maze.num_cols-1):
                maze.grid[i][j].remove_walls(i, j+1)
                maze.grid[i][j+1].remove_walls(i, j)
            break

        # repeat the following for all cells in rows
        for j in range(maze.num_cols):
            # check if we are in the last column
            if j == maze.num_cols - 1:
                # remove only the top wall for this cell
                maze.grid[i][j].remove_walls(i+1, j)
                maze.grid[i+1][j].remove_walls(i, j)
                continue

            # randomly choose between 0 and 1.
            # if we get 0, remove top wall; otherwise remove right wall
            remove_top = random.choice([True, False])

            # if we chose to remove top wall
            if remove_top:
                maze.grid[i][j].remove_walls(i+1, j)
                maze.grid[i+1][j].remove_walls(i, j)
            else:
                maze.grid[i][j].remove_walls(i, j+1)
                maze.grid[i][j+1].remove_walls(i, j)

    print("Number of moves performed: {}".format(maze.num_cols * maze.num_rows))
    print("Execution time for algorithm: {:.4f}".format(time.time() - time_start))

    # choose the entry and exit coordinates
    maze.grid[maze.entry_coor[0]][maze.entry_coor[1]].set_as_entry_exit("entry", maze.num_rows-1, maze.num_cols-1)
    maze.grid[maze.exit_coor[0]][maze.exit_coor[1]].set_as_entry_exit("exit", maze.num_rows-1, maze.num_cols-1)

    # create a path for animating the maze creation using a binary tree
    path = list()
    visit_counter = 0
    visited = list()
    k_curr, l_curr = (maze.num_rows-1, maze.num_cols-1)
    path.append((k_curr, l_curr))

    begin_time = time.time()

    while visit_counter < maze.grid_size:  # While there are unvisited cells
        possible_neighbours = list()

        try:
            if not maze.grid[k_curr-1][l_curr].visited and k_curr != 0:
                if not maze.grid[k_curr][l_curr].is_walls_between(maze.grid[k_curr-1][l_curr]):
                    possible_neighbours.append((k_curr-1, l_curr))
        except:
            pass

        try:
            if not maze.grid[k_curr][l_curr-1].visited and l_curr != 0:
                if not maze.grid[k_curr][l_curr].is_walls_between(maze.grid[k_curr][l_curr-1]):
                    possible_neighbours.append((k_curr, l_curr-1))
        except:
            pass

        if possible_neighbours:
            k_next, l_next = possible_neighbours[0]
            path.append(possible_neighbours[0])
            visited.append((k_curr, l_curr))
            maze.grid[k_next][l_next].visited = True
            visit_counter += 1
            k_curr, l_curr = k_next, l_next
        else:
            if visited:
                k_curr, l_curr = visited.pop()
                path.append((k_curr, l_curr))
            else:
                break

    for row in maze.grid:
        for cell in row:
            cell.visited = False

    print(f"Generating path for maze took {time.time() - begin_time}s.")
    maze.generation_path = path

def display_maze(maze):
    num_rows, num_cols = maze.num_rows, maze.num_cols
    grid = np.zeros((num_rows*2+1, num_cols*2+1))

    for i in range(num_rows):
        for j in range(num_cols):
            if not maze.grid[i][j].walls['N']:
                grid[i*2, j*2+1] = 1
            if not maze.grid[i][j].walls['S']:
                grid[i*2+2, j*2+1] = 1
            if not maze.grid[i][j].walls['E']:
                grid[i*2+1, j*2+2] = 1
            if not maze.grid[i][j].walls['W']:
                grid[i*2+1, j*2] = 1
            if maze.grid[i][j].entry_exit:
                grid[i*2+1, j*2+1] = 2
            else:
                grid[i*2+1, j*2+1] = 1

    plt.imshow(grid, cmap="hot")
    plt.show()

# Create and display a maze using Depth-First Search algorithm
maze = Maze(num_rows=10, num_cols=10)
depth_first_recursive_backtracker(maze, (0, 0))
display_maze(maze)

# Create and display a maze using Binary Tree algorithm
maze = Maze(num_rows=10, num_cols=10)
binary_tree(maze, (0, 0))
display_maze(maze)
