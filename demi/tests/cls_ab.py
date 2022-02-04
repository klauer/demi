from __future__ import annotations


class A:
    "A Docstring"
    a_value = "A"

    def afunc(self) -> str:
        return "A"

    def superfunc(self: A) -> int:
        b = ord(self.afunc())
        a = b + 2
        return a


class B(A):
    "B Docstring"
    b_value = "B"

    def bfunc(self) -> str:
        return "B"

    def superfunc(self) -> int:
        return super().superfunc() + 1


class C(B):
    "C Docstring"
    c_value = "C"

    def cfunc(self) -> str:
        return "C"

    def superfunc(self) -> int:
        # Comment
        return super().superfunc() + 2
