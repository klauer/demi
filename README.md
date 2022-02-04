demi - de-multiple-inheritance; squash class inheritance
========================================================

Did you make a mistake in designing a mess of classes with multiple inheritance
everywhere?

Do you want to understand deeply nested class hierarchies without navigating
many different source code files?

Will you avoid making such mistakes in the future?

... well, my answers to the above questions are "yes :(", "yes!", and "probably
not", respectively.

In any case -

This repo is a proof-of-concept of a tool that may have the potential to help
with the above.  It probably won't help you just yet.

Requirements
------------

* Python 3.9+ (for ``ast.unparse``)

Usage
-----

Install

```bash
$ pip install demi
```

Try with a provided test class:
```bash
$ demi import demi.tests.cls_ab.C
```

Or preferrably, reformat with black at the same time:
```bash
$ demi import demi.tests.cls_ab.C |black -
```

```python
class C:
    """A Docstring

    B Docstring

    C Docstring"""

    a_value = "A"

    def afunc(self) -> str:
        return "A"

    def superfunc(self) -> int:
        def _super_A() -> int:
            b = ord(self.afunc())
            a = b + 2
            return a

        return _super_A() + 1

    b_value = "B"
    c_value = "C"

    def bfunc(self) -> str:
        return "B"

    def cfunc(self) -> str:
        return "C"
```

Or maybe try something from the standard library:
```
demi import argparse.BooleanOptionalAction
demi import tkinter.Widget
...
```

Maybe TODO
----------

* Properties have the same name twice, so only the setter gets shown
* Switch to redbaron to retain comments?
* Automatically invoke black to reformat the output?
* Test suite?
* For returns at the end of the method, simplify code?
* Consider astor for Python below 3.9?
