"""
# Definition for a QuadTree node.
class Node:
    def __init__(self, val, isLeaf, topLeft, topRight, bottomLeft, bottomRight):
        self.val = val
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight
"""

# Given a n * n matrix grid of 0's and 1's only. We want to represent grid with a Quad-Tree.

# Return the root of the Quad-Tree representing grid.

# A Quad-Tree is a tree data structure in which each internal node has exactly four children. Besides, each node has two attributes:

# val: True if the node represents a grid of 1's or False if the node represents a grid of 0's. Notice that you can assign the val to True or False when isLeaf is False, and both are accepted in the answer.
# isLeaf: True if the node is a leaf node on the tree or False if the node has four children.

class Solution:
    def construct(self, grid: List[List[int]]) -> 'Node':
        def is_same_value(x, y, length):
            for i in range(x, x + length):
                for j in range(y, y + length):
                    if grid[i][j] != grid[x][y]:
                        return False
            return True
        
        def construct_helper(x, y, length):
            if is_same_value(x, y, length):
                return Node(grid[x][y] == 1, True, None, None, None, None)
            elif length == 1:
                return Node(grid[x][y] == 1, True, None, None, None, None)
            half = length // 2
            return Node(False, False, 
                        construct_helper(x, y, half),
                        construct_helper(x, y + half, half),
                        construct_helper(x + half, y, half),
                        construct_helper(x + half, y + half, half))
        
        return construct_helper(0, 0, len(grid))