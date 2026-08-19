FROM python:3.11-slim

# git: ingestion/clone.py shells out to the real `git` binary directly
# (clone/pull), not a Python git library -- it has to be on PATH.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY scripts ./scripts
COPY streamlit_app ./streamlit_app

EXPOSE 8000
CMD ["uvicorn", "codeseek.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
