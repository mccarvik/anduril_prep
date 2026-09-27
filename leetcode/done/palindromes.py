class Solution:
    def countSubstrings(self, s: str) -> int:
        
        pals = []

        for subs_len in range(1,len(s)+1):
            for ind in range(len(s)):
                if ind + subs_len > len(s):
                    break
                check_str = s[ind:ind+subs_len]
                if self.check_palindrome(check_str):
                    pals.append(check_str)
        
        print(pals)
        return len(pals)

    
    def check_palindrome(self, s):
        for ind in range(int(len(s)/2)):
            if s[ind] != s[-(ind+1)]:
                return False
        return True