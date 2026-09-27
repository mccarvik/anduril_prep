class Solution:
    def getYDiff(self, a, b):
        # Returns the delta Y.
        return a[1] - b[1]

    def getXDiff(self, a, b):
        # Returns the delta X.
        return a[0] - b[0]

    def checkStraightLine(self, coordinates):
        deltaY = self.getYDiff(coordinates[1], coordinates[0])
        deltaX = self.getXDiff(coordinates[1], coordinates[0])

        for i in range(2, len(coordinates)):
            # Check if the slope between points 0 and i is the same as between 0 and 1.
            if (deltaY * self.getXDiff(coordinates[i], coordinates[0])
                    != deltaX * self.getYDiff(coordinates[i], coordinates[0])):
                return False
        return True