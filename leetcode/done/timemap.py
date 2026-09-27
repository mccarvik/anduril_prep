class TimeMap:

    def __init__(self):
        self.master_dict = {}
        

    def set(self, key: str, value: str, timestamp: int) -> None:
        if key in self.master_dict:
            inner_dict = self.master_dict[key]
            inner_dict[timestamp] = value
            self.master_dict[key] = inner_dict
        else:
            inner_dict = {}
            inner_dict[timestamp] = value
            self.master_dict[key] = inner_dict
        

    def get(self, key: str, timestamp: int) -> str:
        for time in range(timestamp, 0, -1):
            try:
                return self.master_dict[key][time]
            except:
                # print(self.master_dict[key])
                # print()
                continue
        if key in self.master_dict:
            return ""
        return None
        


# Your TimeMap object will be instantiated and called as such:
# obj = TimeMap()
# obj.set(key,value,timestamp)
# param_2 = obj.get(key,timestamp)