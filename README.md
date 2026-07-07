# LOFOP

**LOFOP** is a modular, enterprise-grade computer vision framework built on PyTorch. It is an
original design informed by the architecture of mature open-source projects (Ultralytics,
MMDetection/MMEngine, Detectron2, OpenCV, PyTorch, Transformers) without deriving from their code.

> Status: early development. Phase 1 (core engine) is implemented; see
> [`docs/architecture.md`](docs/architecture.md) for the full roadmap.

## Design goals

- **Modular** -- every subsystem (core, data, models, training, inference, deployment) works
  independently and communicates through small, explicit contracts.
- **Declarative** -- experiments are YAML configs instantiated through registries; behavior lives
  in registered components, configs stay diffable data.
- **Extensible** -- new tasks, models, and integrations arrive as plugins and registry entries,
  never as changes to the framework core.
- **Production-ready** -- typed, tested, logged, and documented from the first commit.

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python 3.9+. The core engine depends only on PyYAML; heavier dependencies (PyTorch,
OpenCV, export toolchains) attach to the subsystems that need them.

## Quick tour of the core engine

```python
from lofop import Config, HUB
from lofop.registries import BACKBONES

@BACKBONES.register()
class RidgeNet:
    def __init__(self, depth: int = 50, width: float = 1.0):
        self.depth, self.width = depth, width

# Declarative instantiation from config data:
backbone = BACKBONES.build({"type": "RidgeNet", "depth": 101})

# Cross-group references use qualified "group/Name" types:
model = HUB.build({"type": "backbone/RidgeNet", "width": 2.0})
```

Configs support inheritance, interpolation, and freezing:

```yaml
# experiment.yaml
extends: [base.yaml]
name: exp42
workdir: runs/${name}
data_root: ${env:DATA_ROOT:/data}
```

```python
cfg = Config.load("experiment.yaml")
cfg.freeze()
```

Lifecycle events and plugins wire everything together:

```python
from lofop import EVENTS, PLUGINS

@EVENTS.on("train.epoch_end")
def report(event):
    print(event["epoch"], event.get("metrics"))

PLUGINS.discover()      # find installed `lofop.plugins` entry points
PLUGINS.activate_all()  # let them register components / subscribe to events
```

## Repository layout

```
lofop/
  core/          # task-agnostic engine: registry, config, events, plugins, logging, exceptions
  registries.py  # default hub + standard component groups (backbone, neck, head, loss, ...)
docs/            # architecture and module documentation
tests/           # pytest suite mirroring the package layout
```

## Development

```bash
python -m pytest                                   # run the test suite
ruff check lofop tests benchmarks                  # lint
python benchmarks/bench_core.py -o report.md       # core engine micro-benchmarks
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
