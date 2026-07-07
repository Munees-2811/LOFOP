"""Component registry system.

The registry is LOFOP's extension point: every pluggable component (backbones,
losses, datasets, hooks, ...) is registered by name into a :class:`Registry`,
and instantiated from declarative config specs via :meth:`Registry.build`.

Design decisions, and why:

* **Flat string names per group, groups collected in a hub.** A single global
  namespace (Detectron2-style) collides quickly; deep hierarchical scopes
  (MMEngine-style) are powerful but hard to reason about. LOFOP uses one
  namespace per component group and a :class:`RegistryHub` that owns the
  groups.
* **Qualified type names for cross-group references.** A spec's ``type`` is
  either ``"Name"`` (resolved in the registry doing the building) or
  ``"group/Name"`` (resolved through the hub). Nested specs are recursively
  built *only* when their ``type`` is qualified, so a plain dict argument that
  happens to contain a ``type`` key is never instantiated by surprise.
* **Registries store factories, not instances.** Anything callable can be
  registered (classes, functions), keeping the system open to functional
  builders and partials.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Callable, TypeVar

from lofop.core.exceptions import BuildError, RegistryError
from lofop.core.logging import get_logger

TYPE_KEY = "type"
_QUALIFIER = "/"

_F = TypeVar("_F", bound=Callable[..., Any])

logger = get_logger(__name__)


class Registry:
    """A named collection of component factories.

    Args:
        name: Group name, e.g. ``"backbone"``. Used in qualified type names
            and error messages.
        hub: Optional :class:`RegistryHub` used to resolve qualified type
            names (``"group/Name"``) during :meth:`build`.
    """

    def __init__(self, name: str, *, hub: RegistryHub | None = None) -> None:
        if not name or _QUALIFIER in name:
            raise RegistryError(
                f"Registry name must be non-empty and must not contain {_QUALIFIER!r}",
                context={"name": name},
            )
        self.name = name
        self._hub = hub
        self._components: dict[str, Callable[..., Any]] = {}

    @property
    def hub(self) -> RegistryHub | None:
        """The hub this registry belongs to, if any."""
        return self._hub

    def register(
        self,
        obj: _F | None = None,
        *,
        name: str | None = None,
        aliases: tuple[str, ...] = (),
        override: bool = False,
    ) -> Any:
        """Register a callable, usable directly or as a decorator.

        Examples::

            @BACKBONES.register()
            class RidgeNet: ...

            @BACKBONES.register(name="ridgenet-s", aliases=("rn-s",))
            class RidgeNetSmall: ...

            BACKBONES.register(build_ridgenet, name="ridgenet_fn")

        Args:
            obj: The callable to register. When omitted, returns a decorator.
            name: Registration name; defaults to ``obj.__name__``.
            aliases: Additional names resolving to the same callable.
            override: Allow replacing an existing registration. Without it a
                duplicate name raises :class:`RegistryError`.

        Returns:
            ``obj`` unchanged (or a decorator when ``obj`` is omitted), so
            classes keep working normally after registration.
        """
        if obj is None:
            def decorator(inner: _F) -> _F:
                self.register(inner, name=name, aliases=aliases, override=override)
                return inner

            return decorator

        if not callable(obj):
            raise RegistryError(
                "Only callables (classes, functions) can be registered",
                context={"registry": self.name, "obj": obj},
            )
        primary = name or getattr(obj, "__name__", None)
        if not primary:
            raise RegistryError(
                "Cannot infer a registration name; pass name= explicitly",
                context={"registry": self.name, "obj": obj},
            )
        for key in (primary, *aliases):
            if _QUALIFIER in key:
                raise RegistryError(
                    f"Registered names must not contain {_QUALIFIER!r}",
                    context={"registry": self.name, "name": key},
                )
            if key in self._components and not override:
                raise RegistryError(
                    "Name already registered; pass override=True to replace it",
                    context={"registry": self.name, "name": key},
                )
            self._components[key] = obj
        logger.debug("Registered %r in registry %r", primary, self.name)
        return obj

    def get(self, name: str) -> Callable[..., Any]:
        """Return the factory registered under ``name``.

        Raises:
            RegistryError: If the name is unknown; the message lists close or
                available names to make config typos easy to fix.
        """
        try:
            return self._components[name]
        except KeyError:
            raise RegistryError(
                f"Unknown component {name!r}",
                context={"registry": self.name, "available": sorted(self._components)},
            ) from None

    def build(self, spec: Mapping[str, Any], /, **overrides: Any) -> Any:
        """Instantiate a component from a declarative spec.

        The spec must contain a ``type`` key naming the factory; all other
        keys become keyword arguments. ``overrides`` take precedence over spec
        keys. Nested mappings whose ``type`` is qualified (``"group/Name"``)
        are recursively built through the hub, including inside lists and
        tuples; unqualified nested mappings pass through untouched.

        Raises:
            BuildError: If the spec is malformed or the factory raises.
        """
        if not isinstance(spec, Mapping):
            raise BuildError(
                "Component spec must be a mapping",
                context={"registry": self.name, "spec": spec},
            )
        merged: dict[str, Any] = {**spec, **overrides}
        type_name = merged.pop(TYPE_KEY, None)
        if not isinstance(type_name, str) or not type_name:
            raise BuildError(
                f"Component spec must contain a non-empty string {TYPE_KEY!r} key",
                context={"registry": self.name, "spec": dict(spec)},
            )
        factory = self._resolve(type_name)
        kwargs = {key: self._materialize(value) for key, value in merged.items()}
        try:
            return factory(**kwargs)
        except BuildError:
            raise
        except Exception as exc:
            raise BuildError(
                f"Failed to build {type_name!r}: {exc}",
                context={"registry": self.name, "type": type_name},
            ) from exc

    def _resolve(self, type_name: str) -> Callable[..., Any]:
        if _QUALIFIER not in type_name:
            return self.get(type_name)
        if self._hub is None:
            raise BuildError(
                "Qualified type names require a registry hub",
                context={"registry": self.name, "type": type_name},
            )
        group, _, name = type_name.partition(_QUALIFIER)
        return self._hub[group].get(name)

    def _materialize(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            type_name = value.get(TYPE_KEY)
            if isinstance(type_name, str) and _QUALIFIER in type_name:
                return self.build(value)
            return value
        if isinstance(value, (list, tuple)):
            return type(value)(self._materialize(item) for item in value)
        return value

    def __contains__(self, name: str) -> bool:
        return name in self._components

    def __iter__(self) -> Iterator[str]:
        return iter(self._components)

    def __len__(self) -> int:
        return len(self._components)

    def __repr__(self) -> str:
        return f"Registry(name={self.name!r}, components={sorted(self._components)})"


class RegistryHub:
    """Owns a set of named registries and resolves qualified specs.

    A hub is the unit of isolation: the framework ships a default hub
    (:data:`lofop.registries.HUB`), while tests or embedded applications can
    create private hubs that cannot leak registrations into each other.
    """

    def __init__(self) -> None:
        self._registries: dict[str, Registry] = {}

    def new(self, name: str) -> Registry:
        """Create and attach a new registry group.

        Raises:
            RegistryError: If the group already exists.
        """
        if name in self._registries:
            raise RegistryError("Registry group already exists", context={"group": name})
        registry = Registry(name, hub=self)
        self._registries[name] = registry
        return registry

    def get_or_new(self, name: str) -> Registry:
        """Return the group named ``name``, creating it if missing.

        Plugins use this to attach components to groups that may or may not
        have been declared yet.
        """
        try:
            return self._registries[name]
        except KeyError:
            return self.new(name)

    def build(self, spec: Mapping[str, Any], /, **overrides: Any) -> Any:
        """Build a spec whose ``type`` must be qualified (``"group/Name"``)."""
        if not isinstance(spec, Mapping):
            raise BuildError("Component spec must be a mapping", context={"spec": spec})
        type_name = spec.get(TYPE_KEY)
        if not isinstance(type_name, str) or _QUALIFIER not in type_name:
            raise BuildError(
                "Hub-level build requires a qualified 'group/Name' type",
                context={"type": type_name},
            )
        group, _, _ = type_name.partition(_QUALIFIER)
        return self[group].build(spec, **overrides)

    def __getitem__(self, name: str) -> Registry:
        try:
            return self._registries[name]
        except KeyError:
            raise RegistryError(
                f"Unknown registry group {name!r}",
                context={"available": sorted(self._registries)},
            ) from None

    def __contains__(self, name: str) -> bool:
        return name in self._registries

    def __iter__(self) -> Iterator[str]:
        return iter(self._registries)

    def __len__(self) -> int:
        return len(self._registries)

    def __repr__(self) -> str:
        return f"RegistryHub(groups={sorted(self._registries)})"
