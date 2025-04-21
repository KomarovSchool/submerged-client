FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the project into the image
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Sync the project into a new environment, using the frozen lockfile
WORKDIR /app
ADD pyproject.toml uv.lock /app
RUN uv sync --frozen
ADD . /app

CMD ["uv", "run", "python", "main.py"]