
class Cell():

    def __init__(self, f, g,h, loc, parent):
        self.f = f
        self.g = g
        self.h = h
        self.loc = loc
        self.parent = parent



    def get_g(self):
        return self.g

    def get_h(self):
        return self.h

    def get_f(self):
        return self.g + self.h

    def get_loc(self):
        return self.loc

    def get_parent(self):
        return self.parent


    def set_g(self,i):
        self.g = i

    def set_h(self,i):
        self.h = i

    def set_f(self,i):
        self.f = i

    def set_loc(self,i):
        self.loc = i

    def set_parent(self,i):
        self.parent = i







