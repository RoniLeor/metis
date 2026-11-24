MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --no-builtin-variables
.DEFAULT_GOAL := help

.PHONY: help sync check check-lint format fix test ground-truth viewer nvidia-data nvidia-tasks nvidia-rows train dojo-source dojo-rows evaluate

help:
	@echo "Targets: sync check check-lint format fix test ground-truth viewer nvidia-data nvidia-tasks nvidia-rows train dojo-source dojo-rows evaluate"
	@echo 'Options: DOJO=resources/data/dojo; NVIDIA=resources/data/nvidia; PYTEST_ARGS="tests/data/labels/export.py -k cli"; TRAIN="epochs=1 limit=200"; EVAL="base=resources/model/weights name=dojo"'

DOJO ?= resources/data/dojo
PYTEST_ARGS ?=
TRAIN ?=
EVAL ?=
TRUTH := $(DOJO)/ground-truth.json
NVIDIA ?= resources/data/nvidia
# The code that the two AgentDojo commands run: when it changes, make runs them again.
AGENTDOJO_CODE := $(wildcard src/metis/data/dojo/extract/*.py src/metis/data/*.py src/metis/data/dojo/models.py src/metis/data/labels/*.py) \
	src/metis/utils.py
NVIDIA_URL := https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1/resolve
NVIDIA_REVISION := d738d4f361cc38bb4d7a42b9066776dade5332f5
NVIDIA_SHA256 := 3329da17564a7eb287e2730fc7d6956e1f4fe51e8950ac4f110b3c37e78cf3b9

# The train extra installs laya and torch, so that type checks and tests cover the laya backend.
sync:
	uv sync --locked --extra train

check: check-lint test

check-lint:
	uv run --no-sync ruff format --check .
	uv run --no-sync ruff check .
	uv run --no-sync pyright

format:
	uv run --no-sync ruff format .

fix: format
	uv run --no-sync ruff check --fix .

test:
	uv run --no-sync pytest -q $(PYTEST_ARGS)

ground-truth: $(TRUTH)

# AgentDojo keeps some lists in sets. A fixed hash seed gives the same output on each run.
$(TRUTH): $(AGENTDOJO_CODE)
	mkdir -p $(DOJO)
	PYTHONHASHSEED=0 uv run --no-sync --with agentdojo==0.1.35 python -m metis.data.dojo.extract.groundtruth $@

viewer: $(TRUTH)
	uv run --no-sync python -m metis.data.labels.export $(DOJO) $(DOJO)/index.html

nvidia-data: $(NVIDIA)/train.jsonl

# A fixed revision and checksum give the same input on each machine.
$(NVIDIA)/train.jsonl:
	mkdir -p $(NVIDIA)
	curl -fsSL -o $@.part $(NVIDIA_URL)/$(NVIDIA_REVISION)/train.jsonl
	echo "$(NVIDIA_SHA256)  $@.part" | shasum -a 256 -c -
	mv $@.part $@

nvidia-tasks: $(NVIDIA)/train.jsonl
	uv run --no-sync python -m metis.data.nvidia.tasks $(NVIDIA) $(NVIDIA)

nvidia-rows: $(NVIDIA)/train.jsonl
	uv run --no-sync python -m metis.data.nvidia.build $(NVIDIA) $(NVIDIA)/rows

train: nvidia-rows
	uv run --no-sync python -m metis.engine.train data=$(NVIDIA)/rows $(TRAIN)

dojo-source: $(DOJO)/rows/source.json

# The AgentDojo strings, clean and with each attack in each reached slot. A fixed hash seed keeps the order.
$(DOJO)/rows/source.json: $(AGENTDOJO_CODE)
	mkdir -p $(DOJO)/rows
	PYTHONHASHSEED=0 uv run --no-sync --with agentdojo==0.1.35 python -m metis.data.dojo.extract.source $@

dojo-rows: $(DOJO)/rows/source.json
	uv run --no-sync python -m metis.data.dojo.build $(DOJO)/rows/source.json $(DOJO)/rows

evaluate: dojo-rows
	uv run --no-sync python -m metis.engine.evaluate data=$(DOJO)/rows $(EVAL)
