class Solution:
    def updateBoard(self, board: List[List[str]], click: List[int]) -> List[List[str]]:
        r, c = click
        if board[r][c] == "M":
            board[r][c] = "X"
            return board

        rows, cols = len(board), len(board[0])
        dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                (0, 1),   (1, -1), (1, 0),  (1, 1)]

        def reveal(r, c):
            if board[r][c] != "E":          # already revealed (or not an empty square)
                return
            neighbors = [(r + dr, c + dc) for dr, dc in dirs
                         if 0 <= r + dr < rows and 0 <= c + dc < cols]
            mines = sum(board[nr][nc] == "M" for nr, nc in neighbors)
            if mines:
                board[r][c] = str(mines)    # numbered square: stop here
            else:
                board[r][c] = "B"           # blank: open all neighbors
                for nr, nc in neighbors:
                    reveal(nr, nc)

        reveal(r, c)
        return board