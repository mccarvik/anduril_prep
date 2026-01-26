class Solution:
    def divisibleTripletCount(self, nums: List[int], d: int) -> int:
        trips = 0
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                for k in range(j+1, len(nums)):
                    if (nums[i]+nums[j]+nums[k]) % d == 0:
                        trips += 1
        return trips