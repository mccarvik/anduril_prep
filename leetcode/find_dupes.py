class Solution:
    def findDuplicate(self, nums: list[int]) -> int:
        newlist = sorted(nums)
        for i in range(len(newlist)+1):
            if newlist[i] == newlist[-1]:
                return newlist[-1]
            if newlist[i] == newlist[i+1]:
                return newlist[i]