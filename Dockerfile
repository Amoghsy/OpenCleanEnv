FROM python:3.10

WORKDIR /app

COPY . .

# Install uv + dependencies
RUN pip install --no-cache-dir uv \
    && uv pip install --system openenv-core[cli] pandas numpy fastapi uvicorn pydantic

EXPOSE 8000

CMD ["uv", "run", "--system", "--project", ".", "server", "--port", "8000"]