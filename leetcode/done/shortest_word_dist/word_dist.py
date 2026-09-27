class Solution:
    def shortestDistance(self, wordsDict: List[str], word1: str, word2: str) -> int:
        idx = 0
        found1 = []
        found2 = []
        for word in wordsDict:
            if word == word1:
                found1.append(idx)
            if word == word2:
                found2.append(idx)
            idx+=1
        
        minword = 3 * 10**4
        for w1 in found1:
            for w2 in found2:
                if abs(w1-w2) < minword:
                    if (w2>w1 and abs(w1-w2) > minword):
                        break
                    minword = abs(w1-w2)
        return minword
