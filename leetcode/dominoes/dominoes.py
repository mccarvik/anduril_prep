class Solution:
    def pushDominoes(self, dominoes: str) -> str:
        dominoes = list(dominoes)
        new_dom = dominoes.copy()
        

        while True:
            for dom in range(len(dominoes)):
                # check ignore
                if dominoes[dom] in ["R", "L"]:
                    continue
                # check left
                if dom != 0 and dominoes[dom-1] == "R":
                    new_dom[dom] = "R"
                
                # check right
                if dom != len(dominoes)-1 and dominoes[dom+1] == "L":
                    if new_dom[dom] == "R":
                        new_dom[dom] = "."
                    else:
                        new_dom[dom] = "L"
            if new_dom == dominoes:
                break
            dominoes = new_dom.copy()


        dominoes = "".join(dominoes)
        return dominoes

        