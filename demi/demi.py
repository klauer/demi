from __future__ import annotations

import ast
import copy
import dataclasses
import inspect
import logging
import textwrap
from typing import Dict, List, Optional, Union, cast

logger = logging.getLogger(__name__)

AnyFunctionDef = Union[ast.FunctionDef, ast.AsyncFunctionDef]
function_def_types = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclasses.dataclass
class ClassDefinition:
    node: ast.ClassDef
    cls: type
    functions: Dict[str, List[AnyFunctionDef]]
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

            mod = ast.parse(cls_source, filename=filename or "demi_unknown.py")
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
                functions=_get_functions(node.body),
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

    def _rewrite_method(
        self,
        supercls: ClassDefinition,
        func_name: str,
    ) -> Optional[List[AnyFunctionDef]]:
        """
        Starting with our subclass, insert the superclass method definition.
        """

        # TODO: not necessarily true; properties, setters?
        supercls_funcs = supercls.functions.get(func_name, None)
        if not supercls_funcs:
            return None

        # TODO might not be correct; how to choose the getter, e.g., of a
        # property?  If instantiable, easy to determine...
        supercls_func = supercls_funcs[-1]

        rewritten_methods = []
        for idx, this_func in enumerate(self.functions[func_name]):
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
            rewriter = DemiMethodRewriter(self, this_func)
            rewritten_method = rewriter.run()
            logger.debug(
                "Rewrote [%s, %s].%s to:\n"
                "%s\n",
                supercls.name,
                self.name,
                rewritten_method.name,
                textwrap.indent(ast.unparse(rewritten_method), "    "),
            )
            assert rewritten_method.name == func_name
            rewritten_methods.append(rewritten_method)

        if rewritten_methods:
            target_indices = [
                supercls.node.body.index(func)
                for func in supercls_funcs
            ]
            for idx, rewritten_method in enumerate(rewritten_methods):
                if idx < len(target_indices):
                    supercls.node.body[target_indices[idx]] = rewritten_method
                else:
                    last_idx = target_indices[-1]
                    supercls.node.body.insert(last_idx + 1, rewritten_method)
            supercls.functions[func_name] = rewritten_methods

        return rewritten_methods or None

    def demi(self) -> ClassDefinition:
        superclasses = self.mro[1:]
        if not superclasses:
            return self

        to_add = []
        supercls = superclasses[0]
        for func_name, this_funcs in list(self.functions.items()):
            rewritten_method = self._rewrite_method(supercls, func_name)
            if rewritten_method is None:
                # TODO try to insert things in best-effort order as a merge
                # new_functions.append(func)
                to_add.extend(this_funcs)

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
            supercls.functions[func.name] = [func]
            supercls.node.body.append(func)

        _replace_base_class(supercls, self)
        supercls.mro = [self] + self.mro[2:]
        supercls.node.name = self.name
        return supercls


def _get_functions(nodes: List[ast.AST]) -> Dict[str, List[AnyFunctionDef]]:
    result = {}  # -> defaultdict(list)
    for node in nodes:
        if isinstance(node, function_def_types):
            result.setdefault(node.name, []).append(node)
    return result


def _get_string_constant(node: ast.AST) -> Optional[str]:
    # -> ast.unparse()
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
    def is_base_class(base):
        # TODO: import scoping and such
        return ast.unparse(base).split(".")[-1] == baseclass.name

    base_idx = [
        idx
        for idx, base in enumerate(subclass.node.bases)
        if is_base_class(base)
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
        self.bases_by_name = {
            supercls.name: supercls
            for supercls in self.cls.mro[1:]
            if self.method_name in supercls.functions
        }
        self.bases = list(self.bases_by_name.values())

    def run(self):
        if not self.bases:
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
                target = self.bases[0]
                super_impls = target.functions.get(self.method_name, None)
                # TOOD: how to choose?
                super_impl = copy.deepcopy(super_impls[-1]) if super_impls else None
                if super_impl is None or self.method_name != func_name:
                    # TODO: function could actually be trying to skip
                    # subclass implementation of unrelated (func_name)
                    outer_call.func.id = f"self.{func_name}"
                elif super_impl not in self._to_insert:
                    outer_call.func.id = f"_super_{target.name}"
                    if super_impl.args.args[0].arg == "self":
                        super_impl.args.args = super_impl.args.args[1:]
                    super_impl.name = outer_call.func.id
                    self._to_insert.append(super_impl)
                    # TODO: better way around this?
                    outer_call.args = node.args
                    outer_call.keywords = node.keywords
                else:
                    raise
                return outer_call

        return node
