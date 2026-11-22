class Solution:
    def findBuildings(self, heights: list[int]) -> list[int]:
        tallest = 0
        builds = []
        for i in range(len(heights), -1, -1):
            if heights[i-1]>tallest:
                builds.append((i-1,heights[i-1]))
                tallest = heights[i-1]
        print(builds)
        first_values = [x[0] for x in sorted(builds, key=lambda x: x[1], reverse=True)]
        return first_values