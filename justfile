set shell := ["bash", "-lc"]

default:
    just --list

init:
    just venv && \
    just sync

venv:
    test -d .venv || uv venv

sync:
    uv sync --all-packages

sync-lock:
    uv sync --locked --all-packages

up:
    uv lock --upgrade

add PKG:
    uv add {{PKG}}

add-dev PKG:
    uv add --dev {{PKG}}

rm PKG:
    uv remove {{PKG}}

rm-dev PKG:
    uv remove --dev {{PKG}}

fmt:
    just sync && \
    uv run ruff format .

fmt-check:
    uv run ruff format --check .

lint:
    uv run ruff check . --fix

lint-check:
    uv run ruff check .

test:
    uv run pytest

ci:
    just venv && \
    just sync-lock && \
    just fmt-check && \
    just lint-check && \
    just test

DATA_CRSP_DATE := "2026_02_13"

find-last D:
    uv run python scripts/tools/find_last.py {{D}}

pl-qf-01 DATE=DATA_CRSP_DATE O="data/pipeline/qf/01_meta" S="src":
    uv run python scripts/pipeline/qf/01_meta.py "data/external/crsp/{{DATE}}" "{{O}}" "{{S}}"
