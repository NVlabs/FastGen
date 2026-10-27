all: help

install-linters: ## install the linters
	pip install ruff==0.6.9 mypy==2.1.0

install: install-linters ## install the development dependencies (alias for install-linters)

mypy: ## smoke-check train.py with mypy (fastgen/ modules are not analysed)
	python3 -m mypy --check-untyped-defs --follow-imports=skip train.py

lint: ## check formatting and lint rules with ruff
	python3 -m ruff format --check
	python3 -m ruff check

format: ## format the code and auto-fix lint rules with ruff
	python3 -m ruff format
	python3 -m ruff check --fix

install-fastgen: ## install FastGen in editable mode
	python3 -m pip install -e .

# Run the suite in three sequential pytest invocations so memory accumulated by
# tests that load large pretrained models (HF caches, CUDA allocator state,
# diffusers/transformers globals) is reclaimed between batches. Each batch
# always runs (so CI gets a complete picture) and the final exit code reflects
# whether any batch failed.
pytest: ## run the test suite in three batches
	@rc=0; \
	ulimit -n 4096 || echo "Could not raise file descriptor limit"; \
	echo "===== batch 1/3: all tests except tests/test_network.py and tests/test_network_fsdp.py ====="; \
	python3 -m pytest \
		--ignore=FASTGEN_OUTPUT --ignore=runs --ignore=tmp \
		--ignore=tests/test_network.py --ignore=tests/test_network_fsdp.py \
		|| rc=$$?; \
	echo "===== batch 2/3: tests/test_network.py ====="; \
	python3 -m pytest tests/test_network.py || rc=$$?; \
	echo "===== batch 3/3: tests/test_network_fsdp.py ====="; \
	python3 -m pytest tests/test_network_fsdp.py || rc=$$?; \
	exit $$rc

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
