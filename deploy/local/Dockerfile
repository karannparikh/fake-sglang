FROM python:3.10-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /opt/fake-sglang
COPY deploy/local/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install pip==25.2 \
    && python -m pip install -r /tmp/requirements.txt \
       --extra-index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY models ./models
COPY tests ./tests
COPY scripts ./scripts
COPY deploy/local ./deploy/local
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -e '.[dev]' \
    && python -c "import sglang; assert sglang.__is_fake_sglang__" \
    && python -m dynamo.frontend --help > /dev/null \
    && python -m dynamo.sglang --help > /dev/null

RUN useradd --create-home --uid 1000 simulator
USER simulator
CMD ["python", "-m", "dynamo.sglang", "--help"]
