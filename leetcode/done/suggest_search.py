class Solution:
    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
        str_arr = [x for x in searchWord]
        ret_arr = []
        for x in range(1,len(str_arr)+1):
            chars = "".join(str_arr[:x])
            this_arr = []
            for word in products:
                if chars == word[:len(chars)]:
                    this_arr.append(word)
            print(this_arr)
            this_arr = sorted(this_arr)[:3]
            ret_arr.append(this_arr)
        return ret_arr