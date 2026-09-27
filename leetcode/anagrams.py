class Solution:
    def groupAnagrams(self, strs: list[str]) -> list[list[str]]:
        buckets = []  # each entry: (key, [originals])
        for s in strs:
            key = sorted(s)
            for k, group in buckets:
                if k == key:
                    group.append(s)
                    break
            else:
                buckets.append((key, [s]))
        return [group for _, group in buckets]