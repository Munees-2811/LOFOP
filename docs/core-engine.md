# Core Engine Reference

`lofop.core` is the task-agnostic foundation of LOFOP. Its only third-party dependency is PyYAML.

## Registry system

```python
from lofop.core import Registry, RegistryHub
from lofop.registries import BACKBONES, HUB   # the default hub and standard groups
```

### Registering components

```python
@BACKBONES.register()                                # name defaults to class name
class RidgeNet: ...

@BACKBONES.register(name="ridgenet-s", aliases=("rn-s",))
class RidgeNetSmall: ...

BACKBONES.register(build_ridgenet, name="ridgenet_fn")   # functions work too
```

Duplicate names raise `RegistryError` unless `override=True`. Names may not contain `/` (reserved
for qualification).

### Building from specs

```python
BACKBONES.build({"type": "RidgeNet", "depth": 101})          # unqualified: this group
HUB.build({"type": "backbone/RidgeNet", "depth": 101})       # qualified: any group
BACKBONES.build(spec, depth=50)                              # kwargs override spec keys
```

**Recursion rule:** inside a spec, a nested mapping (including inside lists/tuples) is built
recursively *iff* its `type` is qualified (`"group/Name"`). Unqualified nested mappings pass
through as plain dicts. Example:

```yaml
type: head/DensePointHead
num_classes: 80
loss: {type: loss/FocalCost, gamma: 2.0}   # built -> FocalCost instance
layout: {type: grid, stride: 8}            # NOT built -> passed as dict
```

Errors: unknown names raise `RegistryError` listing available names; malformed specs and factory
exceptions raise `BuildError` with the failing type attached (original exception chained).

### Custom hubs

```python
hub = RegistryHub()
models = hub.new("model")          # raises if the group exists
extras = hub.get_or_new("model")   # idempotent; what plugins should use
```

## Configuration

```python
from lofop.core import Config
```

- Attribute + item access: `cfg.model.depth`, `cfg["model"]["depth"]`.
- Dotted paths: `cfg.select("model.backbone.depth")`, `cfg.select("a.b", default=None)`,
  `cfg.update_path("data.workers", 8)`.
- `cfg.merge(other)` deep-merges in place; `cfg.freeze()` makes the tree read-only
  (`ConfigError` on mutation); `cfg.to_dict()` / `cfg.dump(path)` export plain data.

### File loading

```yaml
# exp.yaml
extends: [base.yaml, schedule.yaml]   # relative to this file; merged in order, child wins
name: exp42
workdir: runs/${name}                 # config reference (textual inside larger strings)
lr: ${base_lr}                        # whole-string reference preserves type (float)
data_root: ${env:DATA_ROOT:/data}     # environment with default
```

`Config.load(path)` applies inheritance, then interpolation (`resolve=False` defers it). Cycles in
`extends` or references, missing files, bad YAML, unknown references, and unset env vars without
defaults all raise `ConfigError` with context.

## Event bus

```python
from lofop.core import EventBus
bus = EventBus()

sub = bus.subscribe("train.epoch_end", handler, priority=10)   # higher runs first
bus.subscribe("train.epoch_end", tracker, once=True)           # auto-removed after one delivery

@bus.on("checkpoint.saved")
def upload(event):
    path = event["path"]              # payload access; event.get("key", default) also works

failures = bus.emit("train.epoch_end", epoch=3, metrics=m)     # [(handler_name, exc), ...]
bus.emit("train.epoch_end", raise_errors=True, epoch=3)        # EventError carrying failures
bus.unsubscribe(sub); bus.clear("topic"); bus.clear()
```

Delivery is synchronous and deterministic: priority descending, then subscription order. Handler
exceptions never propagate unless `raise_errors=True`.

## Plugins

```python
from lofop.registries import PLUGINS   # default manager bound to HUB + EVENTS

def setup(context):                     # PluginContext(hub=..., events=...)
    context.hub.get_or_new("backbone").register(MyNet, name="mynet")
    context.events.subscribe("train.start", my_handler)

PLUGINS.add("my-plugin", setup)         # programmatic
PLUGINS.discover()                      # scan `lofop.plugins` entry points (metadata only)
PLUGINS.activate("my-plugin")           # runs setup; idempotent
PLUGINS.activate_all()
```

Packages ship plugins by declaring an entry point:

```toml
[project.entry-points."lofop.plugins"]
my-plugin = "my_pkg.lofop_plugin:setup"
```

Plugin code is imported only at activation. Load/setup failures raise `PluginError` (plugin name
and source in the message) and leave the plugin inactive.

## Logging

```python
from lofop.core import configure_logging, get_logger

configure_logging(level="INFO", log_file="runs/exp42/lofop.log")   # opt-in app setup
log = get_logger(__name__)                                          # "lofop.<module>"
```

- All framework loggers live under the `"lofop"` root; libraries embedding LOFOP simply do not
  call `configure_logging` and attach their own handlers instead.
- `configure_logging` is idempotent (`force=True` to re-apply), reads `LOFOP_LOG_LEVEL` when no
  level is given, uses Rich formatting when installed and stderr is a TTY (`use_rich` to force),
  and file handlers always capture DEBUG regardless of console level.

## Exceptions

```
LofopError                # base; catch-all for framework failures
├── ConfigError
├── RegistryError
├── BuildError
├── PluginError
└── EventError            # .failures -> [(handler_name, exception), ...]
```

All accept `context={...}`; pairs render into the message and stay available on `.context` for
structured error reporting.
