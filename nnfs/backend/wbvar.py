import numpy as np

# helper class so I replace "weight * a; bias * a"
# with: "wbvar * a"
class WBVar:
    def __init__(self, w_arr: np.ndarray, bias_arr: np.ndarray):
        self.w = w_arr
        self.b = bias_arr

    def __add__(self, other):
        if isinstance(other, WBVar):
            return WBVar(self.w + other.w, self.b + other.b)
        else:
            return WBVar(self.w + other, self.b + other)

    def __neg__(self):
        return WBVar(-self.w, -self.b)

    def __sub__(self, other):
        return self.__add__(other.__neg__())

    def __mul__(self, other):
        if isinstance(other, WBVar):
            return WBVar(self.w * other.w, self.b * other.b)
        else:
            return WBVar(self.w * other, self.b * other)

    def __div__(self, other):
        if isinstance(other, WBVar):
            return WBVar(self.w / other.w, self.b / other.b)
        else:
            return WBVar(self.w / other, self.b / other)

    def __truediv__(self, other):
        return self.__div__(other)

    def __pow__(self, other):
        if isinstance(other, WBVar):
            return WBVar(self.w ** other.w, self.b ** other.b)
        else:
            return WBVar(self.w ** other, self.b ** other)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return self.__add__(other.__neg__())

    def __rmul__(self, other):
        return self * other

    def __rdiv__(self, other):
        return self / other

    def __rtruediv__(self, other):
        return self / other

    def __rpow__(self, other):
        return self ** other

    def __iadd__(self, other):
        if isinstance(other, WBVar):
            self.w += other.w
            self.b += other.b
        else:
            self.w += other
            self.b += other

    def __isub__(self, other):
        if isinstance(other, WBVar):
            self.w -= other.w
            self.b -= other.b
        else:
            self.w -= other
            self.b -= other

    def __imul__(self, other):
        if isinstance(other, WBVar):
            self.w *= other.w
            self.b *= other.b
        else:
            self.w *= other
            self.b *= other

    def __idiv__(self, other):
        if isinstance(other, WBVar):
            self.w /= other.w
            self.b /= other.b
        else:
            self.w /= other
            self.b /= other

    def __itruediv__(self, other):
        self.__idiv__(other)

    def __ipow__(self, other):
        if isinstance(other, WBVar):
            self.w **= other.w
            self.b **= other.b
        else:
            self.w **= other
            self.b **= other
