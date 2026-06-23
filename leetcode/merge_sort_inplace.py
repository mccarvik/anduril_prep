class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        nums1[m:] = nums2          # replace the n zero slots with nums2
        nums1.sort()               # in-place sort