# test prep

class Circle:
    """ circle class """
    def __init__(self, radius):
        """
        constructor
        """
        self.rad = radius
        self.pi = 3.141

    def area(self):
        """
        calc the area
        """

        area = self.pi * self.rad**2
        return area
    
class Rectangle:
    """ rectangle class """
    def __init__(self, width, height):
        """
        constructor
        """
        self.w = width
        self.h = height

    def area(self):
        """
        calc the area
        """
        area = self.w * self.h
        return area
    
class Square:
    """ square class """
    def __init__(self, side):
        """
        constructor
        """
        self.s = side
        
    def area(self):
        """
        calc the area
        """
        area = self.s**2
        return area