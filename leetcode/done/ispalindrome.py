class Solution:
    def isPalindrome(self, x: int) -> bool:
        x_str = list(str(x))
        print(x_str)
        for x in range(len(x_str)//2):
            if x_str[x] != x_str[-x-1]:
                return False
        
        return True