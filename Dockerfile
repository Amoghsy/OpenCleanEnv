ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app/env

# -------- ENV --------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV ENABLE_WEB_INTERFACE=true
ENV PATH="/app/env/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# -------- COPY FILES --------
COPY . /app/env

# -------- CREATE VENV --------
RUN python -m venv /app/env/.venv
RUN /app/env/.venv/bin/pip install --upgrade pip setuptools wheel

# -------- INSTALL DEPENDENCIES (CRITICAL) --------
RUN /app/env/.venv/bin/pip install --no-cache-dir \
    "openenv-core[core]>=0.2.3" \
    fastapi \
    uvicorn \
    pandas \
    numpy \
    requests \
    openai

# -------- INSTALL PROJECT --------
RUN /app/env/.venv/bin/pip install --no-deps -e /app/env

# -------- PORT --------
EXPOSE 7860

# -------- RUN SERVER --------
CMD ["/app/env/.venv/bin/uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]