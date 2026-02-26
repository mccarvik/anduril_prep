from typing import List


class Solution:
    def largestIsland(self, grid: List[List[int]]) -> int:
        island_sizes = {}
        island_id = 2

        for cur_row in range(len(grid)):
            for cur_col in range(len(grid[0])):
                if grid[cur_row][cur_col] == 1:
                    island_sizes[island_id] = self.explore_island(grid, island_id, cur_row, cur_col)
                    island_id += 1

        # if we have no islands, return 1
        # we can flip any square and get 1
        if not island_sizes:
            return 1


        # if there is only one island
        if len(island_sizes) == 1:
            island_id -= 1
            if island_sizes[island_id] == len(grid) * len(grid[0]):
                return island_sizes[island_id]
            else:
                return island_sizes[island_id] + 1
        
        max_isle_size = 1

        # ok base cases done
        # now we look where the conversion will be most impactful
        for cur_row in range(len(grid)):
            for cur_col in range(len(grid)):

                if grid[cur_row][cur_col] == 0:
                    cur_isle_size = 1
                    neighs = set()

                    # check down
                    if (cur_row+1 < len(grid) and grid[cur_row+1][cur_col] > 1):
                        neighs.add(grid[cur_row+1][cur_col])
                    
                    # check up
                    if (cur_row-1 >= 0 and grid[cur_row-1][cur_col] > 1):
                        neighs.add(grid[cur_row-1][cur_col])

                    # check right
                    if (cur_col+1 < len(grid[0]) and grid[cur_row][cur_col+1] > 1):
                        neighs.add(grid[cur_row][cur_col+1])

                    # check left
                    if (cur_col-1 >= 0 and grid[cur_row][cur_col-1] > 1):
                        neighs.add(grid[cur_row][cur_col-1])
                    
                    # sum the sizes of all our neighbors
                    for island_id in neighs:
                        cur_isle_size += island_sizes[island_id]

                    # for each cell, get the value of flipping it and then see if this is the best flip
                    max_isle_size = max(max_isle_size, cur_isle_size)
        return max_isle_size


    def explore_island(self, grid, island_id, cur_row, cur_col):
        # check edges
        if (cur_row < 0 or cur_row>=len(grid) or cur_col < 0 or cur_col >= len(grid[0]) or grid[cur_row][cur_col] != 1):
            return 0
        
        # get current id
        grid[cur_row][cur_col] = island_id

        return 1 + self.explore_island(grid, island_id, cur_row+1, cur_col) + \
                self.explore_island(grid, island_id, cur_row-1, cur_col) + \
                self.explore_island(grid, island_id, cur_row, cur_col+1) + \
                self.explore_island(grid, island_id, cur_row, cur_col-1)
        # add 1 to total
        # check down
        # check top
        # check right
        # check left

# Time complexity: O(n×m)

# The algorithm consists of two main phases. 
# In the first phase, we iterate over every cell in the grid to identify and 
# mark islands using a Depth-First Search (DFS) approach. 
# \During this process, each cell is visited at most once, 
# ensuring that the DFS traversal contributes O(n×m) to the time complexity.

# Space complexity: O(n×m)

# The space complexity is primarily determined by the recursion stack used during 
# the DFS traversal and the storage required for the unordered map that keeps 
# track of island sizes. In the worst case, the recursion depth of the 
# DFS can be O(n×m) if the entire grid forms a single large island. 
# The unordered map stores the sizes of all islands, 
# and in the worst case, the number of islands can be proportional
# to the number of cells, contributing O(n×m) to the space complexity.