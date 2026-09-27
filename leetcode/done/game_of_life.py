from typing import List

class Solution:
    def gameOfLife(self, board: List[List[int]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        rows = len(board)
        cols = len(board[0])
        new_b = [[0] * cols for _ in range(rows)]

        for x in range(rows):
            for y in range(cols):
                neighs = self.neighbors(board, x, y)
                
                # Apply Game of Life rules
                if board[x][y] == 1:  # Currently alive
                    if neighs == 2 or neighs == 3:
                        new_b[x][y] = 1  # Lives on
                    else:
                        new_b[x][y] = 0  # Dies
                else:  # Currently dead
                    if neighs == 3:
                        new_b[x][y] = 1  # Becomes alive
                    else:
                        new_b[x][y] = 0  # Stays dead
        
        # Modify board in-place as required
        for x in range(rows):
            for y in range(cols):
                board[x][y] = new_b[x][y]
        
        return new_b
                

    def neighbors(self, mat, x, y):
        neighs = 0
        if x!=0 and y!=0 and mat[x-1][y-1] == 1:
            neighs+=1

        if y!=0 and mat[x][y-1] == 1:
            neighs+=1

        if x!=len(mat)-1 and y!=0 and mat[x+1][y-1] == 1:
            neighs+=1
        
        if x!=0 and mat[x-1][y] == 1:
            neighs+=1
        
        if x!=len(mat)-1 and mat[x+1][y] == 1:
            neighs+=1
        
        if x!=0 and y!=len(mat[x])-1 and mat[x-1][y+1] == 1:
            neighs+=1

        if y!=len(mat[x])-1 and mat[x][y+1] == 1:
            neighs+=1

        if x!=len(mat)-1 and y!=len(mat[x])-1 and mat[x+1][y+1] == 1:
            neighs+=1
        
        return neighs
        