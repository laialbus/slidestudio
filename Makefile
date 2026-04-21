.PHONY: install test lint

install:
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "Error: no virtual environment active. Run 'source venv/bin/activate' first."; \
		exit 1; \
	fi
	
	pip install -r requirements.txt
	pip install -r requirements-dev.txt -q

test:
	pytest tests/

lint:
	@echo "Lint not yet configured."
