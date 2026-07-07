"""Tests for lofop.core.registry."""

import pytest

from lofop.core.exceptions import BuildError, RegistryError
from lofop.core.registry import Registry, RegistryHub


class Widget:
    def __init__(self, size=1, child=None, children=()):
        self.size = size
        self.child = child
        self.children = list(children)


class TestRegistry:
    def test_register_decorator_and_get(self):
        reg = Registry("thing")

        @reg.register()
        class Foo:
            pass

        assert reg.get("Foo") is Foo
        assert "Foo" in reg
        assert len(reg) == 1

    def test_register_with_name_and_aliases(self):
        reg = Registry("thing")
        reg.register(Widget, name="widget", aliases=("w",))
        assert reg.get("widget") is Widget
        assert reg.get("w") is Widget

    def test_duplicate_name_rejected_unless_override(self):
        reg = Registry("thing")
        reg.register(Widget, name="widget")
        with pytest.raises(RegistryError):
            reg.register(Widget, name="widget")
        reg.register(Widget, name="widget", override=True)

    def test_register_rejects_non_callable(self):
        reg = Registry("thing")
        with pytest.raises(RegistryError):
            reg.register(42, name="answer")

    def test_registered_name_cannot_contain_qualifier(self):
        reg = Registry("thing")
        with pytest.raises(RegistryError):
            reg.register(Widget, name="a/b")

    def test_unknown_component_lists_available(self):
        reg = Registry("thing")
        reg.register(Widget, name="widget")
        with pytest.raises(RegistryError) as excinfo:
            reg.get("wdget")
        assert "widget" in str(excinfo.value)

    def test_build_basic_and_overrides(self):
        reg = Registry("thing")
        reg.register(Widget, name="widget")
        built = reg.build({"type": "widget", "size": 3})
        assert isinstance(built, Widget) and built.size == 3
        assert reg.build({"type": "widget", "size": 3}, size=7).size == 7

    def test_build_requires_type(self):
        reg = Registry("thing")
        with pytest.raises(BuildError):
            reg.build({"size": 3})

    def test_build_wraps_factory_errors(self):
        reg = Registry("thing")

        def exploding(**kwargs):
            raise ValueError("boom")

        reg.register(exploding, name="bad")
        with pytest.raises(BuildError) as excinfo:
            reg.build({"type": "bad"})
        assert "boom" in str(excinfo.value)

    def test_registry_name_validation(self):
        with pytest.raises(RegistryError):
            Registry("")
        with pytest.raises(RegistryError):
            Registry("a/b")


class TestQualifiedBuild:
    def make_hub(self):
        hub = RegistryHub()
        hub.new("model").register(Widget, name="widget")
        hub.new("loss").register(Widget, name="cost")
        return hub

    def test_nested_qualified_spec_is_built(self):
        hub = self.make_hub()
        built = hub["model"].build(
            {"type": "widget", "child": {"type": "loss/cost", "size": 5}}
        )
        assert isinstance(built.child, Widget) and built.child.size == 5

    def test_nested_unqualified_mapping_passes_through(self):
        hub = self.make_hub()
        built = hub["model"].build({"type": "widget", "child": {"type": "cost", "size": 5}})
        assert built.child == {"type": "cost", "size": 5}

    def test_nested_specs_inside_lists(self):
        hub = self.make_hub()
        built = hub["model"].build(
            {"type": "widget", "children": [{"type": "loss/cost", "size": 2}, {"plain": 1}]}
        )
        assert isinstance(built.children[0], Widget)
        assert built.children[1] == {"plain": 1}

    def test_qualified_type_without_hub_fails(self):
        reg = Registry("standalone")
        reg.register(Widget, name="widget")
        with pytest.raises(BuildError):
            reg.build({"type": "widget", "child": {"type": "loss/cost"}})


class TestRegistryHub:
    def test_new_and_getitem(self):
        hub = RegistryHub()
        reg = hub.new("backbone")
        assert hub["backbone"] is reg
        assert reg.hub is hub
        assert "backbone" in hub

    def test_duplicate_group_rejected(self):
        hub = RegistryHub()
        hub.new("backbone")
        with pytest.raises(RegistryError):
            hub.new("backbone")

    def test_get_or_new_is_idempotent(self):
        hub = RegistryHub()
        assert hub.get_or_new("x") is hub.get_or_new("x")

    def test_unknown_group(self):
        hub = RegistryHub()
        with pytest.raises(RegistryError):
            hub["nope"]

    def test_hub_build_requires_qualified_type(self):
        hub = RegistryHub()
        hub.new("model").register(Widget, name="widget")
        assert isinstance(hub.build({"type": "model/widget"}), Widget)
        with pytest.raises(BuildError):
            hub.build({"type": "widget"})


def test_default_hub_declares_standard_groups():
    from lofop.registries import BACKBONES, HUB

    for group in (
        "model", "backbone", "neck", "head", "loss", "metric",
        "optimizer", "scheduler", "dataset", "transform", "hook",
    ):
        assert group in HUB
    assert HUB["backbone"] is BACKBONES
