class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        diff = len(s) - len(a) + 1
        pot_inds_i = [i for i in range(diff)]
        pot2_i = []
        print(pot_inds_i)
        for iii in pot_inds_i:
            if s[iii:iii + len(a)] == a:
                pot2_i.append(iii)
        
        diff2 = len(s) - len(a)
        pot_inds_j = [j for j in range(diff2)]
        pot2_j = []
        for jjj in pot_inds_j:
            if s[jjj:jjj + len(b)] == b:
                pot2_j.append(jjj)
        
        rets = []
        for i in pot2_i:
            for j in pot2_j:
                if abs(j-i) <= k and i not in rets:
                    rets.append(i)
        
        return sorted(rets)