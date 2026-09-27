# Convert a non-negative integer num to its English words representation.

class Solution:
    def numberToWords(self, num: int) -> str:
        if num == 0:
            return "Zero"
        thousands = ["", "Thousand", "Million", "Billion"]
        self.hundreds = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
        self.tens = ["", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
        self.teens = ["Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
        words = []
        for i in range(len(thousands) - 1, -1, -1):
            chunk = num // 1000 ** i % 1000
            if chunk == 0:
                continue
            piece = self.helper(chunk)
            if thousands[i]:
                piece += " " + thousands[i]
            words.append(piece)
        return " ".join(words)

    def helper(self, num: int) -> str:
        words = []
        if num >= 100:
            words.append(self.hundreds[num // 100])
            words.append("Hundred")
            num %= 100
        if num >= 20:
            words.append(self.tens[num // 10])
            num %= 10
        elif 11 <= num <= 19:
            words.append(self.teens[num - 11])
            num = 0
        elif num == 10:
            words.append(self.tens[1])
            num = 0
        if num:
            words.append(self.hundreds[num])
        return " ".join(words)
