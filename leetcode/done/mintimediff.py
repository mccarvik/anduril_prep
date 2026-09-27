# Given a list of 24-hour clock time points in "HH:MM" format, return the minimum 
# minutes difference between any two time-points in the list.

class Solution:
    def findMinDifference(self, timePoints: List[str]) -> int:
        # Convert time points to minutes since midnight
        minutes = [self.timeToMinutes(time) for time in timePoints]
        minutes.sort()
        
        # Calculate the minimum difference between consecutive time points
        min_diff = float('inf')
        for i in range(len(minutes) - 1):
            min_diff = min(min_diff, minutes[i + 1] - minutes[i])
        
        # Handle wrap-around case (e.g., 23:59 to 00:00)
        min_diff = min(min_diff, 1440 - minutes[-1] + minutes[0])
        
        return min_diff

    # Helper function to convert time to minutes since midnight
    def timeToMinutes(self, time: str) -> int:
        hours, minutes = map(int, time.split(':'))
        return hours * 60 + minutes