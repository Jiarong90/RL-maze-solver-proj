from collections import deque

# Find guaranteed shortest path with BFS. To use as part of convergence evaluation check
def bfs_shortest_path_length(maze, start, goal):
    rows, cols = maze.shape
    visited = set()
    queue = deque([(start, 0)])  # (position, distance)
    while queue:
        (r, c), dist = queue.popleft() # Pops from queue in FIFO order
        if (r, c) == goal:
            return dist
        # Positions it can move to
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            # New row, column
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr, nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), dist + 1))
    return float('inf')  # no path

def bfs_compute(environment, start_cell):
    # Convert (col, row) to (row, col) with ::-1 as stated in maze.py
    start = start_cell[::-1]
    goal = environment.goal[::-1]
    return bfs_shortest_path_length(environment.maze, start, goal)