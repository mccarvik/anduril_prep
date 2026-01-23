class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        courses = [x for x in range(0, numCourses)]
        for preq in prerequisites:
            for ind in range(len(courses)):
                if courses[ind] == preq[0]:
                    ind1 = ind
                if courses[ind] == preq[1]:
                    ind2 = ind
            
            if ind1 > ind2:
                continue
            else:
                courses.pop(ind1)
                courses.insert(ind2, ind1)
        return courses