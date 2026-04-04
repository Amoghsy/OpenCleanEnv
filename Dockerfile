ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app/env

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PATH="/app/env/.venv/bin:$PATH"
ENV ENABLE_WEB_INTERFACE=true

COPY pyproject.toml /app/env/
COPY openenv.yaml /app/env/
COPY server/requirements.txt /app/env/server/

RUN python -m venv /app/env/.venv
RUN /app/env/.venv/bin/pip install --upgrade pip setuptools wheel

# 🔥 install uv + deps
RUN /app/env/.venv/bin/pip install uv
RUN /app/env/.venv/bin/pip install \
    openenv-core[core]>=0.2.3 \
    fastapi \
    uvicorn \
    pandas \
    numpy \
    requests \
    openai

COPY . /app/env
RUN /app/env/.venv/bin/pip install --no-deps -e /app/env

EXPOSE 8000

CMD ["/app/env/.venv/bin/uv", "run", "server", "--port", "8000"]