class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        
        visited = [False for _ in range(len(s))]
        swaps = {}
        strg = list(s)

        # add all the potential swap options
        for s, d in pairs:
            try:
                swaps[s].append(d)
            except:
                swaps[s] = [d]
            try:
                swaps[d].append(s)
            except:
                swaps[d] = [s]

        def dfs(strg, ind, chars, indices):
            print(str(ind) + " ind")
            print(str(chars) + " chars")
            print(str(strg) + " strg")
            print(str(indices) + " indices")
            
            if visited[ind]:
                return
            chars.append(strg[ind])
            indices.append(ind)
            visited[ind] = True

            for neigh in swaps[ind]:
                dfs(strg, neigh, chars, indices)
            
        for i in range(len(strg)):
            if not visited[i]:
                chars = []
                indices = []
                dfs(strg, i, chars, indices)
                chars = sorted(chars)
                indices = sorted(indices)
                for c, i in zip(chars, indices):
                    strg[i] = c
        
        return "".join(strg)


