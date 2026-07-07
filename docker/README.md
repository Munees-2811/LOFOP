# Docker images

Three image variants are provided; pick the one that matches your runtime.

| Image | Base | Use for | Approx. focus |
|---|---|---|---|
| `Dockerfile-cpu` | `python:3.11-slim` + torch CPU wheels | CPU inference, dataset tooling, CI | portability |
| `Dockerfile-gpu` | `pytorch/pytorch` CUDA 12.4 runtime | training, GPU inference | throughput |
| `Dockerfile-onnx` | `python:3.11-slim` + ONNX Runtime | edge/cloud serving of exported models | smallest footprint |

All images install LOFOP with its CLI as the entrypoint, run as a non-root
`lofop` user, and honor `LOFOP_LOG_LEVEL`.

## Build

Always build from the **repository root** (the Dockerfiles copy `lofop/` and
`pyproject.toml`):

```bash
docker build -f docker/Dockerfile-cpu  -t lofop:cpu  .
docker build -f docker/Dockerfile-gpu  -t lofop:gpu  .
docker build -f docker/Dockerfile-onnx -t lofop:onnx .
```

The root `Dockerfile` is an alias for the CPU variant, so `docker build .`
also works.

## Run

The entrypoint is the `lofop` CLI; any CLI invocation works directly:

```bash
docker run --rm lofop:cpu version

# Dataset tools: mount your data and work against /data paths.
docker run --rm -v /path/to/data:/data lofop:cpu \
    dataset convert --from coco --source /data/instances.json \
    --to yolo --target /data/yolo --image-root /data/images

docker run --rm -v /path/to/data:/data lofop:cpu \
    dataset validate --format yolo --source /data/yolo

# GPU (requires the NVIDIA Container Toolkit):
docker run --rm --gpus all -v /path/to/data:/data lofop:gpu \
    dataset stats --format coco --source /data/instances.json
```

For an interactive shell instead of the CLI:

```bash
docker run --rm -it --entrypoint bash lofop:cpu
```

## Notes

- Dependency layers are ordered so editing framework source only rebuilds the
  final `pip install .` layer.
- Exact torch/CUDA pins live in the Dockerfiles; bump them deliberately and
  rebuild all variants together so CI results stay comparable.
