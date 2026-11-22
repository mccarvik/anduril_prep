# class Solution:
#     def findRadius(self, houses: list[int], heaters: list[int]) -> int:
#         radius = 0
#         for h in houses:
#             nearest = float('inf')
#             for w in heaters:
#                 nearest = min(nearest, abs(h - w))
#             radius = max(radius, nearest)
#         return radius


from bisect import bisect_left

class Solution:
    def findRadius(self, houses: list[int], heaters: list[int]) -> int:
        heaters.sort()
        radius = 0
        for h in houses:
            i = bisect_left(heaters, h)
            left = h - heaters[i - 1] if i > 0 else float('inf')
            right = heaters[i] - h if i < len(heaters) else float('inf')
            radius = max(radius, min(left, right))
        return radius