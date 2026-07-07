# Default image = the CPU variant; see docker/README.md for all variants.
# Build: docker build -t lofop .

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LOFOP_LOG_LEVEL=INFO

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 lofop
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY lofop/ lofop/
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install .

# Compile the native C++ ops (NMS/IoU fast path), then drop the toolchain.
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && python -c "from lofop.ops.native import build_native; build_native()" \
    && apt-get purge -y g++ && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

USER lofop
ENTRYPOINT ["lofop"]
CMD ["--help"]
