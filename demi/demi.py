from __future__ import annotations

import ast
import copy
import dataclasses
import inspect
import logging
import textwrap
from typing import List, Optional, Union, cast

logger = logging.getLogger(__name__)

AnyFunctionDef = Union[ast.FunctionDef, ast.AsyncFunctionDef]
function_def_types = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclasses.dataclass
class ClassDefinition:
    node: ast.ClassDef
    cls: type
    functions: dict[str, AnyFunctionDef]
    mro: list[ClassDefinition]

    def to_code(self) -> str:
        return ast.unparse(self.node)

    @property
    def name(self) -> str:
        return self.node.name

    @classmethod
    def from_class(cls, klass: type) -> ClassDefinition:
        definitions = []
        for part in klass.mro():
            try:
                cls_source = inspect.getsource(part)
                filename = inspect.getsourcefile(part)
            except TypeError as ex:
                if "built-in" in str(ex):
                    continue
                raise

            mod = ast.parse(cls_source, filename=filename)
            node, = mod.body

            assert isinstance(node, ast.ClassDef)
            if part is klass:
                part_mro = []
            else:
                # TODO: perf, this isn't necessary to redo
                part_mro = ClassDefinition.from_class(part).mro

            defn = ClassDefinition(
                node=node,
                cls=part,
                functions={
                    func.name: func
                    for func in node.body
                    if isinstance(func, function_def_types)
                },
                mro=part_mro,
                # source_filename=...
            )
            definitions.append(defn)

        definitions[0].mro = definitions
        logger.debug(
            "Class %s has mro: %s",
            definitions[0].name,
            [defn.name for defn in definitions],
        )
        return definitions[0]

    def demi_full(self, debug: bool = False) -> ClassDefinition:
        result = self
        while len(result.mro) > 1:
            logger.debug(
                "Demi step; remaining mro: %s\n%s",
                [cls.name for cls in result.mro],
                textwrap.indent(ast.unparse(result.node), "   "),
            )
            result = result.demi()
        return result

    def demi(self) -> ClassDefinition:
        superclasses = self.mro[1:]
        if not superclasses:
            return self

        to_add = []
        supercls = superclasses[0]
        for func_name, this_func in list(self.functions.items()):
            supercls_func = supercls.functions.get(func_name, None)
            if supercls_func:
                logger.debug(
                    "Combining superclass %s.%s:\n"
                    "%s\n"
                    "With %s.%s:\n"
                    "%s\n",
                    supercls.name,
                    supercls_func.name,
                    textwrap.indent(ast.unparse(supercls_func), "    "),
                    self.name,
                    this_func.name,
                    textwrap.indent(ast.unparse(this_func), "    "),
                )
                old_idx = supercls.node.body.index(supercls_func)
                rewriter = DemiMethodRewriter(self, this_func)
                rewritten_method = rewriter.run()
                supercls.node.body[old_idx] = rewritten_method
                supercls.functions[func_name] = rewritten_method
                logger.debug(
                    "Rewrote [%s, %s].%s to:\n"
                    "%s\n",
                    supercls.name,
                    self.name,
                    rewritten_method.name,
                    textwrap.indent(ast.unparse(rewritten_method), "    "),
                )
                assert rewritten_method.name == func_name
            else:
                # TODO try to insert things in best-effort order as a merge
                # new_functions.append(func)
                to_add.append(this_func)

        for idx, node in enumerate(self.node.body):
            const = _get_string_constant(node)
            if const is None:
                break

            self.node.body.pop(0)
            _insert_docstring(supercls.node.body, node)

        for node in self.node.body:
            if not isinstance(node, function_def_types):
                supercls.node.body.append(node)

        for func in to_add:
            supercls.functions[func.name] = func
            supercls.node.body.append(func)

        _replace_base_class(supercls, self)
        supercls.mro = [self] + self.mro[2:]
        supercls.node.name = self.name
        return supercls


def _get_string_constant(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Expr):
        if isinstance(node.value, ast.Constant):
            const = node.value.value
            if isinstance(const, str):
                return const
    return None


def _insert_docstring(body: List[ast.AST], doc: ast.Expression):
    for node in body[:1]:
        if _get_string_constant(node) is not None:
            node.value.value = "\n\n".join(
                (node.value.value, _get_string_constant(doc))
            )
            return

    body.insert(0, doc)


def _replace_base_class(baseclass: ClassDefinition, subclass: ClassDefinition):
    """Remove ``baseclass`` from ``subclass`` and add in its subclasses."""
    # TODO: import scoping and such
    base_idx = [
        idx
        for idx, base in enumerate(subclass.node.bases)
        if getattr(base, "id") == baseclass.name
    ]

    if base_idx:
        for idx in base_idx:
            subclass.node.bases.pop(idx)
        base_idx = base_idx[0]
    else:
        base_idx = 0

    to_add = [
        base
        for base in baseclass.node.bases
        if base not in subclass.node.bases
    ]
    for new_cls in reversed(to_add):
        subclass.node.bases.insert(base_idx, new_cls)


class DemiMethodRewriter(ast.NodeTransformer):
    def __init__(
        self,
        cls: ClassDefinition,
        method_node: AnyFunctionDef,
    ):
        self.cls = cls
        self.method_node = method_node
        self.method_name = method_node.name
        self.base_targets_by_name = {
            supercls.name: supercls
            for supercls in self.cls.mro[1:]
            if self.method_name in supercls.functions
        }
        self.base_targets = list(self.base_targets_by_name.values())

    def run(self):
        # from_func = self.method_node
        # to_func = None
        if not self.base_targets:
            return self.method_node

        self._to_insert = []
        for child in self.method_node.body:
            self.visit(child)
        for item in reversed(self._to_insert):
            self.method_node.body.insert(0, item)

        return self.method_node

    def visit_Call(self, node: ast.Call) -> Union[ast.Call, list[ast.AST]]:
        if not isinstance(node.func, ast.Attribute):
            return node

        func_name = node.func.attr
        outer_call = node.func.value
        if outer_call is not None and isinstance(outer_call, ast.Call):
            outer_call = cast(ast.Call, outer_call)
            outer_func = getattr(outer_call.func, "id", None)
            if outer_func == "super":
                target = self.base_targets[0]
                to_insert = copy.deepcopy(target.functions.get(self.method_name, None))
                if to_insert is None or self.method_name != func_name:
                    # TODO: function could actually be trying to skip
                    # subclass implementation of unrelated (func_name)
                    outer_call.func.id = f"self.{func_name}"
                elif to_insert not in self._to_insert:
                    outer_call.func.id = f"_super_{target.name}"
                    if to_insert.args.args[0].arg == "self":
                        to_insert.args.args = to_insert.args.args[1:]
                    to_insert.name = outer_call.func.id
                    self._to_insert.append(to_insert)
                    # TODO: better way around this?
                    outer_call.args = node.args
                    outer_call.keywords = node.keywords
                else:
                    raise
                return outer_call

        return node


def test():
    from tests.cls_ab import C
    logging.basicConfig(level="DEBUG")

    defn_c = ClassDefinition.from_class(C)
    print("Original")
    print(defn_c.to_code())

    print("\n\nStep 1")
    defn_c = defn_c.demi()
    print(defn_c.to_code())

    print("\n\nStep 2")
    defn_c = defn_c.demi()
    print(defn_c.to_code())


def test_state():
    from pcdsdevices.state import StateRecordPositioner
    logging.basicConfig(level="DEBUG")

    defn_c = ClassDefinition.from_class(StateRecordPositioner)
    defn_c = defn_c.demi_full(debug=True)
    print(defn_c.to_code())


def test_ophyd():
    from ophyd.areadetector.filestore_mixins import \
        FileStoreHDF5SingleIterativeWrite
    logging.basicConfig(level="DEBUG")

    defn_c = ClassDefinition.from_class(FileStoreHDF5SingleIterativeWrite)
    defn_c = defn_c.demi_full(debug=True)
    print(defn_c.to_code())


if __name__ == "__main__":
    # test()
    # test_state()
    test_ophyd()
