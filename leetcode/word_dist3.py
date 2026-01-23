class Solution:
    def shortestWordDistance(self, wordsDict: List[str], word1: str, word2: str) -> int:
        w1 = []
        w2 = []
        for ind in range(len(wordsDict)):
            if wordsDict[ind] == word1:
                w1.append(ind)
            if wordsDict[ind] == word2:
                w2.append(ind)
            
        dist = 1000000
        for ww1 in w1:
            for ww2 in w2:
                if abs(ww1-ww2) < dist and ww1!=ww2:
                    dist = abs(ww1-ww2)
        
        return dist