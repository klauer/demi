"""
`demi import` will import a class and run de-MI on it.
"""

import argparse
import importlib
import logging
import sys
import textwrap

from .. import demi

DESCRIPTION = __doc__

logger = logging.getLogger(__name__)


def build_arg_parser(argparser=None):
    if argparser is None:
        argparser = argparse.ArgumentParser()

    argparser.description = DESCRIPTION
    argparser.formatter_class = argparse.RawTextHelpFormatter

    argparser.add_argument(
        "class_name",
        type=str,
        help="The class name to import and run 'demi' on",
    )

    argparser.add_argument(
        "-n", "--iterations",
        default=0,
        type=int,
        help="The number of iterations to run (otherwise entirely)",
    )

    return argparser


def import_helper(class_name: str) -> type:
    """
    Get a class object from a full qualified class name.

    Parameters
    ----------
    class_name : str
        The module path to find the class e.g.
        ``"pcdsdevices.device_types.IPM"``

    Returns
    -------
    cls : type
        The class referred to by the input string.

    Raises
    ------
    ImportError
        If unable to import the class or module.
    """
    mod, cls = class_name.rsplit('.', 1)
    # Import the module if not already present
    # Otherwise use the stashed version in sys.modules
    if mod in sys.modules:
        logger.debug("Using previously imported version of %s", mod)
        mod = sys.modules[mod]
    else:
        logger.debug("Importing %s", mod)
        mod = importlib.import_module(mod)

    try:
        return getattr(mod, cls)
    except AttributeError as exc:
        raise ImportError(
            f"Unable to import {cls} from {mod.__name__}"
        ) from exc


def main(class_name: str, *, iterations: int = 0):
    logger.debug("Class name: %s", class_name)
    cls = import_helper(class_name)
    logger.debug(
        "Class module=%s name=%s mro=%s", cls.__module__, cls.__name__, cls.mro()
    )

    defn = demi.ClassDefinition.from_class(cls)
    if iterations <= 0:
        defn = defn.demi_full()
    else:
        for iter in range(iterations):
            logger.debug(
                "Iteration %d / %d:\n\n%s",
                iter + 1, iterations,
                textwrap.indent(defn.to_code(), "    ")
            )
            defn = defn.demi()

    print(defn.to_code())
