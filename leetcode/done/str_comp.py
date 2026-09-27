# Given an array of characters chars, compress it using the following algorithm:
# Begin with an empty string s. For each group of consecutive repeating characters in chars:
# If the group's length is 1, append the character to s.
# Otherwise, append the character followed by the group's length.
# The compressed string s should not be returned separately, but instead be stored in the input character array chars. Note that group lengths that are 10 or longer will be split into multiple characters.
# After you are done modifying the input array, return the new length of the array.
# You must write an algorithm that uses only constant extra space.

from typing import List


class Solution:
    def compress(self, chars: List[str]) -> int:
        i = 0
        j = 0
        while i < len(chars):
            count = 1
            while i + 1 < len(chars) and chars[i] == chars[i + 1]:
                i += 1
                count += 1
            chars[j] = chars[i]
            j += 1
            if count > 1:
                for c in str(count):
                    chars[j] = c
                    j += 1
            i += 1
        return j
