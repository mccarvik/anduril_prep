class Solution:
    def orangesRotting(self, grid: list[list[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        minutes = 0

        while self.checkfresh(grid):
            grid_temp = [row[:] for row in grid]  # real copy, not an alias
            changed = False

            for m in range(rows):
                for n in range(cols):
                    if grid[m][n] == 2:
                        for dm, dn in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            r, c = m + dm, n + dn
                            if 0 <= r < rows and 0 <= c < cols and grid[r][c] == 1:
                                grid_temp[r][c] = 2
                                changed = True

            if not changed:      # fresh oranges remain but nothing can reach them
                return -1
            grid = grid_temp
            minutes += 1

        return minutes

    def checkfresh(self, grid: list[list[int]]) -> bool:
        for row in grid:
            if 1 in row:
                return True
        return False