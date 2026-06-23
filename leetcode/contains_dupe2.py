
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        for i in range(len(nums)):
            for j in range(max(i - k, 0), i):
                if nums[i] == nums[j]:
                    return True
        return False