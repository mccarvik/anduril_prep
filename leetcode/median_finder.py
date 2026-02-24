import heapq

class MedianFinder:

    def __init__(self):
        self.lower_half = [] # store the negatives, max heap
        self.upper_half = [] #min side
        

    def addNum(self, num: int) -> None:

        # If it's smaller than or equal to the top of lowerHalf, push it into lowerHalf.
        heapq.heappush(self.lower_half, -num)
        # Otherwise, push it into upperHalf

        # Balance largest of lowerHalf into upperHalf
        # Rebalance the heaps so that:
        # |size(lowerHalf) - size(upperHalf)| <= 1
        heapq.heappush(self.upper_half, -heapq.heappop(self.lower_half))

        # Maintain size property
        if len(self.upper_half) > len(self.lower_half):
            heapq.heappush(self.lower_half, -heapq.heappop(self.upper_half))


    def findMedian(self) -> float:
        # To find the median:
        # If both heaps have the same size → median = average of tops.
        # Otherwise → median = top of the larger heap.
        if len(self.lower_half) == 0:
            return 0
        if len(self.lower_half) > len(self.upper_half):
            return -self.lower_half[0]
        return (-self.lower_half[0] + self.upper_half[0]) / 2.0


# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()

# Time complexity:
# O(log n) for addNum (due to heap insertion)
# O(1) for findMedian
# Space complexity:
# O(n) to store the numbers in heaps