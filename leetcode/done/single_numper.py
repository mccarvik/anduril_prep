class Solution:
    def singleNumber(self, nums: List[int]) -> int:

        seen = []
        print(nums)
        for i in nums:
            print(i)
            if i in seen:
                seen.remove(i)
            else:
                seen.append(i)
            # print(seen)
        return seen[0]