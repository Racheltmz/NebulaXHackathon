"""AST-based public-signature contract enforcement (adapted from the
openevolve-scientist template). Rejects any candidate that renames, removes,
or changes the parameters of the required `Model.__init__/fit/predict`
interface. Candidates may add new methods/helpers freely; they may not
change the frozen interface the harness depends on."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


def argument_contract(arguments: ast.arguments) -> dict[str, Any]:
    return {
        "posonly": [item.arg for item in arguments.posonlyargs],
        "positional": [item.arg for item in arguments.args],
        "vararg": arguments.vararg.arg if arguments.vararg else None,
        "kwonly": [item.arg for item in arguments.kwonlyargs],
        "kwarg": arguments.kwarg.arg if arguments.kwarg else None,
        "defaults": [ast.dump(item, include_attributes=False) for item in arguments.defaults],
        "kw_defaults": [
            ast.dump(item, include_attributes=False) if item is not None else None
            for item in arguments.kw_defaults
        ],
    }


def public_contract(program_path: str | Path) -> dict[str, Any]:
    tree = ast.parse(Path(program_path).read_text(encoding="utf-8"))
    contract: dict[str, Any] = {"functions": {}, "classes": {}}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            contract["functions"][node.name] = argument_contract(node.args)
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            methods: dict[str, Any] = {}
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                    not child.name.startswith("_") or child.name == "__init__"
                ):
                    methods[child.name] = argument_contract(child.args)
            contract["classes"][node.name] = methods
    return contract


def validate_public_contract(program_path: str, initial_program_path: str) -> list[str]:
    expected = public_contract(initial_program_path)
    candidate = public_contract(program_path)
    violations: list[str] = []
    for name, signature in expected["functions"].items():
        if name not in candidate["functions"]:
            violations.append(f"missing public function {name}")
        elif candidate["functions"][name] != signature:
            violations.append(f"changed signature for function {name}")
    for class_name, methods in expected["classes"].items():
        candidate_methods = candidate["classes"].get(class_name)
        if candidate_methods is None:
            violations.append(f"missing public class {class_name}")
            continue
        for method_name, signature in methods.items():
            if method_name not in candidate_methods:
                violations.append(f"missing method {class_name}.{method_name}")
            elif candidate_methods[method_name] != signature:
                violations.append(f"changed signature for {class_name}.{method_name}")
    return violations


FORBIDDEN_IMPORTS = {
    "socket", "urllib", "http", "requests", "ftplib", "smtplib", "telnetlib",
    "subprocess", "os.system", "shutil", "ctypes", "multiprocessing",
}


def static_safety_scan(program_path: str) -> list[str]:
    """Best-effort static check for obviously suspicious imports/calls, on top
    of (not instead of) the network-disabled sandbox. Flags rather than a
    silent bypass -- the sandbox is the real control; this just catches
    obviously bad-faith code early with a clear reason."""
    warnings: list[str] = []
    tree = ast.parse(Path(program_path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_IMPORTS:
                    warnings.append(f"suspicious import: {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in FORBIDDEN_IMPORTS:
                warnings.append(f"suspicious import: {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            # only flag bare-name calls (eval(...), exec(...)) -- an
            # ast.Attribute match here would also catch harmless method
            # calls that happen to share a name, e.g. torch's model.eval()
            # to toggle evaluation mode, which is not the builtin eval().
            if node.func.id in {"eval", "exec", "__import__", "open"}:
                warnings.append(f"suspicious call: {node.func.id}")
    return warnings
