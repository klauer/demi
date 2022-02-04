import pytest

from .. import demi
from . import cls_ab

_classes = [
    cls_ab.A,
    cls_ab.B,
    cls_ab.C,
]

all_classes_param = pytest.mark.parametrize(
    "cls",
    [
        pytest.param(
            cls,
            id=f"{cls.__module__}.{cls.__name__}"
        )
        for cls in _classes
    ]
)


@all_classes_param
def test_smoke(cls: type):
    defn = demi.ClassDefinition.from_class(cls)
    print("Input")
    print(defn.to_code())
    defn = defn.demi_full()
    print("Output")
    print(defn.to_code())
