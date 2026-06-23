class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        for iii in ransomNote:
            if iii not in magazine:
                return False
            else:
                magazine = magazine.replace(iii, "", 1)
    
        return True