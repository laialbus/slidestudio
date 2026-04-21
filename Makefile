.PHONY: install test lint

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	pytest tests/

lint:
	@echo "Lint not yet configured."
