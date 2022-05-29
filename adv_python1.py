class Dachshund:
    def __init__(self, color: str):
        self.color = color
    def show_info(self):
        print (f"This is dachshund, my color is {self.color}")

class Poodle:
    def __init(self, color: str):
        self.color = color
    def show_info(self):
        print (f"This is poolde, my color is {self.color}")

bim = Dachshund("black")
bim.show_info()

# super().__init__() makes the child class
# inherits all parents class methods

# above code can be written as

class Dog:
    def __init__(self, type_: str, color: str):
        self.type = type_
        self.color = color
    def show_info(self):
        print (f"This is {self.type}, color is {self.color}")

class Dachshund(Dog):
    def __init__(self, color: str):
        super().__init__(type_='Dachshund',
                         color=color)
class Poodle(Dog):
    def __init__(self, color: str):
        super().__init__(type_='Poodle',
                         color=color)

bim = Dachshund("black")
bim.show_info()

coco = Poodle("brown")
coco.show_info()


# same methods, but different implementations
# thus we need abstract methods

from abc import ABC, abstractmethod

class Animal(ABC):
    def __init__(self, name: str):
        self.name = name
        super().__init__()
    @abstractmethod
    def make_sound(self):
        pass

class Dog(Animal):
    def make_sound(self):
        print (f'{self.name} make woof')

Dog("Pepper").make_sound()

# class method
# returns constructor of the class
# can be access without creating an object of class

class Solver:
    def __init__(self, nums: list):
        self.nums = nums
    @classmethod
    def get_even(cls, nums: list):
        return cls([num/2 for num in nums])
    def show(self):
        print (self.nums)

nums = [2, 3, 5, 6]
Solver(nums).show()
Solver.get_even(nums).show()

# getattr
# when you want default value

class Food:
    def __init__(self, name: str, color: str):
        self.name = name
        self.color = color

apple = Food("apple", "red")
print ("Color of apple is ",
       getattr(apple, "color", "yellow"))

# static method
# does not access any properties of the class
import re

class ProcessText:
    def __init__(self,
                 text_column: str):
        self.text_column = text_column
    @staticmethod
    def remote_URL(sample: str) -> str:
        return re.sub(r"http\S+", "", sample)

text = ProcessText.remote_URL("go to https://www.google.com")


# property decorator
# getter and setter

class Fruit:
    def __init__(self, name: str,
                 color: str):
        self._name = name
        self._color = color

    @property
    def color(self):
        print ("The color of the fruit is ")
        return self._color
    @color.setter
    def color(self, value):
        print ("Setting value")
        if self._color is None:
            self.color = value
        else:
            print ("Cannot change values")

fruit = Fruit("apple", "red")
fruit.color

