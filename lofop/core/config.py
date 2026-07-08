"""Configuration engine.

LOFOP configs are plain YAML files promoted to :class:`Config` objects that
support attribute access, dotted-path lookup, deep merging, inheritance, and
value interpolation. Design decisions:

* **YAML data, not executable config.** Python-file configs are expressive
  but hard to validate, diff, and ship to production systems that must treat
  configs as data. LOFOP configs are declarative; behavior lives in
  registered components.
* **Inheritance via ``extends``.** A config may list one or more parent
  files (paths relative to the child file). Parents are deep-merged in order,
  then the child is merged on top. This mirrors how experiment configs are
  written in practice: a base recipe plus small overrides.
* **Interpolation after merging.** ``${a.b.c}`` references another config
  value; ``${env:VAR}`` / ``${env:VAR:default}`` reads the process
  environment. Resolution happens on the fully merged tree so overrides
  compose predictably.
* **Freezing.** ``freeze()`` makes a config (recursively) read-only, which
  catches silent mutation bugs once training starts.
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Iterator, Mapping, MutableMapping
from pathlib import Path
from typing import Any

import yaml

from lofop.core.exceptions import ConfigError

EXTENDS_KEY = "extends"
_INTERP_RE = re.compile(r"\$\{([^${}]+)\}")
_MISSING = object()


class Config(MutableMapping[str, Any]):
    """A nested, attribute-accessible configuration mapping.

    Args:
        data: Initial contents. Nested mappings are converted to ``Config``
            recursively; the input is deep-copied so the caller's data is
            never aliased.
    """

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        object.__setattr__(self, "_data", {})
        object.__setattr__(self, "_frozen", False)
        if data:
            for key, value in data.items():
                self[key] = copy.deepcopy(value)

    # -- construction ------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path, *, resolve: bool = True) -> Config:
        """Load a YAML config file, applying ``extends`` inheritance.

        Args:
            path: Path to the YAML file.
            resolve: Interpolate ``${...}`` references after merging.

        Raises:
            ConfigError: On missing files, invalid YAML, non-mapping roots,
                or cyclic ``extends`` chains.
        """
        config = cls(cls._load_tree(Path(path), ancestors=()))
        if resolve:
            config.resolve()
        return config

    @classmethod
    def _load_tree(cls, path: Path, *, ancestors: tuple[Path, ...]) -> dict[str, Any]:
        path = path.resolve()
        if path in ancestors:
            chain = " -> ".join(str(p) for p in (*ancestors, path))
            raise ConfigError("Cyclic 'extends' chain", context={"chain": chain})
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            msg = f"Cannot read config file: {exc}"
            raise ConfigError(msg, context={"path": str(path)}) from exc
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML: {exc}", context={"path": str(path)}) from exc
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            ctx = {"path": str(path), "got": type(raw).__name__}
            raise ConfigError("Config root must be a mapping", context=ctx)
        parents = raw.pop(EXTENDS_KEY, [])
        if isinstance(parents, (str, Path)):
            parents = [parents]
        if not isinstance(parents, list):
            raise ConfigError(
                f"{EXTENDS_KEY!r} must be a path or list of paths", context={"path": str(path)}
            )
        merged: dict[str, Any] = {}
        for parent in parents:
            parent_tree = cls._load_tree(path.parent / parent, ancestors=(*ancestors, path))
            merged = _deep_merge(merged, parent_tree)
        return _deep_merge(merged, raw)

    # -- mapping protocol ----------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError:
            raise KeyError(key) from None

    def __setitem__(self, key: str, value: Any) -> None:
        self._check_mutable()
        if isinstance(value, Mapping) and not isinstance(value, Config):
            value = Config(value)
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        self._check_mutable()
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # -- attribute access ------------------------------------------------

    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(f"Config has no key {key!r}") from None

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no key {key!r}") from None

    # -- dotted-path access ------------------------------------------------

    def select(self, dotted: str, default: Any = _MISSING) -> Any:
        """Return the value at a dotted path, e.g. ``"model.backbone.depth"``.

        Args:
            dotted: Dot-separated key path.
            default: Returned when the path is missing; without it a missing
                path raises :class:`ConfigError`.
        """
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                if default is _MISSING:
                    raise ConfigError("Missing config path", context={"path": dotted})
                return default
        return node

    def update_path(self, dotted: str, value: Any) -> None:
        """Set the value at a dotted path, creating intermediate mappings."""
        self._check_mutable()
        parts = dotted.split(".")
        node: Config = self
        for part in parts[:-1]:
            child = node._data.get(part)
            if not isinstance(child, Config):
                child = Config()
                node[part] = child
            node = child
        node[parts[-1]] = value

    # -- merging / freezing / export ---------------------------------------

    def merge(self, other: Mapping[str, Any]) -> Config:
        """Deep-merge ``other`` on top of this config, in place."""
        self._check_mutable()
        for key, value in other.items():
            current = self._data.get(key)
            if isinstance(current, Config) and isinstance(value, Mapping):
                current.merge(value)
            else:
                self[key] = copy.deepcopy(value)
        return self

    def freeze(self) -> Config:
        """Recursively make this config read-only."""
        object.__setattr__(self, "_frozen", True)
        for value in self._data.values():
            if isinstance(value, Config):
                value.freeze()
        return self

    @property
    def is_frozen(self) -> bool:
        """Whether this config rejects mutation."""
        return self._frozen

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, deep-copied ``dict`` tree."""
        result: dict[str, Any] = {}
        for key, value in self._data.items():
            result[key] = value.to_dict() if isinstance(value, Config) else copy.deepcopy(value)
        return result

    def dump(self, path: str | Path) -> None:
        """Serialize the config to a YAML file (parents created as needed)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )

    # -- interpolation ------------------------------------------------------

    def resolve(self) -> Config:
        """Interpolate ``${path}`` and ``${env:VAR[:default]}`` references.

        References resolve against the root of this config. A reference that
        covers a whole string value preserves the referenced value's type;
        references embedded in larger strings are substituted textually.

        Raises:
            ConfigError: On unknown references, unset environment variables
                without defaults, or cyclic references.
        """
        self._check_mutable()
        self._resolve_node(self, root=self, active=set())
        return self

    def _resolve_node(self, node: Config, *, root: Config, active: set[str]) -> None:
        for key, value in list(node._data.items()):
            if isinstance(value, Config):
                self._resolve_node(value, root=root, active=active)
            elif isinstance(value, list):
                node._data[key] = [
                    _resolve_value(item, root=root, active=active) for item in value
                ]
            else:
                node._data[key] = _resolve_value(value, root=root, active=active)

    def _check_mutable(self) -> None:
        if self._frozen:
            raise ConfigError("Config is frozen and cannot be modified")

    def __repr__(self) -> str:
        return f"Config({self.to_dict()!r})"


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_value(value: Any, *, root: Config, active: set[str]) -> Any:
    if not isinstance(value, str):
        return value
    full = _INTERP_RE.fullmatch(value)
    if full:
        resolved = _resolve_reference(full.group(1), root=root, active=active)
        return copy.deepcopy(resolved)

    def substitute(match: re.Match[str]) -> str:
        return str(_resolve_reference(match.group(1), root=root, active=active))

    return _INTERP_RE.sub(substitute, value)


def _resolve_reference(expr: str, *, root: Config, active: set[str]) -> Any:
    expr = expr.strip()
    if expr.startswith("env:"):
        _, _, rest = expr.partition(":")
        var, sep, default = rest.partition(":")
        result = os.environ.get(var)
        if result is None:
            if sep:
                return default
            raise ConfigError("Environment variable is not set", context={"variable": var})
        return result
    if expr in active:
        raise ConfigError("Cyclic config reference", context={"reference": expr})
    target = root.select(expr, default=_MISSING_REF)
    if target is _MISSING_REF:
        raise ConfigError("Unknown config reference", context={"reference": expr})
    active.add(expr)
    try:
        return _resolve_value(target, root=root, active=active)
    finally:
        active.discard(expr)


_MISSING_REF = object()
