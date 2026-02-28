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

pl-qf:
    just pl-qf-01 && \
    just pl-qf-02 && \
    just pl-qf-03 && \
    just pl-qf-04 && \
    just pl-qf-05 && \
    just pl-qf-06 && \
    just pl-qf-07 && \
    just pl-qf-08 && \
    just pl-qf-09

pl-qf-01 DATE=DATA_CRSP_DATE O="data/pipeline/qf/01_meta" S="src":
    uv run python scripts/pipeline/qf/01_meta.py "data/external/crsp/{{DATE}}" "{{O}}" "{{S}}"

pl-qf-02 I="data/pipeline/qf/01_meta" O="data/pipeline/qf/02_universe" S="src":
    uv run python scripts/pipeline/qf/02_universe.py "$(just find-last {{I}})" "{{O}}" "{{S}}" data/pipeline/qf/00_raw/{{DATA_CRSP_DATE}}

pl-qf-03 I="data/pipeline/qf/01_meta" O="data/pipeline/qf/03_features" S="src":
    uv run python scripts/pipeline/qf/03_features.py "$(just find-last {{I}})" "{{O}}" "{{S}}" data/pipeline/qf/00_raw/{{DATA_CRSP_DATE}}

pl-qf-04 I="data/pipeline/qf/03_features" O="data/pipeline/qf/04_portfolio_sort" S="src":
    uv run python scripts/pipeline/qf/04_portfolio_sort.py "$(just find-last {{I}})" "{{O}}" "{{S}}"

pl-qf-05 I="data/pipeline/qf/03_features" O="data/pipeline/qf/05_cs_ols" S="src":
    uv run python scripts/pipeline/qf/05_cs_ols.py "$(just find-last {{I}})" "{{O}}" "{{S}}"

pl-qf-06 I4="data/pipeline/qf/04_portfolio_sort" I5="data/pipeline/qf/05_cs_ols" O="data/pipeline/qf/06_report" S="src":
    uv run python scripts/pipeline/qf/06_report.py "$(just find-last {{I4}})" "$(just find-last {{I5}})" "{{O}}" "{{S}}"

pl-qf-07 I6="data/pipeline/qf/06_report" O="data/pipeline/qf/07_compare" S="src":
    uv run python scripts/pipeline/qf/07_compare.py "{{I6}}" "{{O}}" "{{S}}"

pl-qf-08 I6="data/pipeline/qf/06_report" O="data/pipeline/qf/08_specs" S="src":
    uv run python scripts/pipeline/qf/08_specs.py "$(just find-last {{I6}})" "{{O}}" "{{S}}"

pl-qf-09 I4="data/pipeline/qf/04_portfolio_sort" I5="data/pipeline/qf/05_cs_ols" O="data/pipeline/qf/09_figures" S="src":
    uv run python scripts/pipeline/qf/09_figures.py "$(just find-last {{I4}})" "$(just find-last {{I5}})" "{{O}}" "{{S}}"

pl-qf-10 I4="data/pipeline/qf/04_portfolio_sort" I5="data/pipeline/qf/05_cs_ols" O="data/pipeline/qf/09_figures" S="src":
    uv run python scripts/pipeline/qf/09_figures.py "$(just find-last {{I4}})" "$(just find-last {{I5}})" "{{O}}" "{{S}}"
