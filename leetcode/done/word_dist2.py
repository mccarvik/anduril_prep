class WordDistance:

    def __init__(self, wordsDict: List[str]):
        self.words = wordsDict
        

    def shortest(self, word1: str, word2: str) -> int:
        if word1 not in self.words or word2 not in self.words:
            return None
        
        min_dist = 10000
        word1s = []
        word2s = []
        for i in range(len(self.words)):
            if self.words[i]==word1:
                word1s.append(i)
            if self.words[i]==word2:
                word2s.append(i)
        
        
        for w1 in word1s:
            for w2 in word2s:
                if abs(w1-w2)<min_dist:
                    min_dist = abs(w1-w2)
        
        return min_dist