## Install Python dependencies
install:
	@echo "Installing python dependencies..."
	pip install poetry
	poetry install

## Activate virtual environment
activate:
	@echo "Activating virtual environment..."
	poetry shell

## Setup project
setup: install activate


## Lint code
lint:
	@echo "Linting code..."
	poetry run ruff check $(SRC) --fix

## Format code using Black + isort
format:
	@echo "🎨 Formatting with black and isort..."
	poetry run black $(SRC)
	poetry run isort $(SRC)

## Check code style without fixing (flake8, mypy, black --check, isort --check)
style-check:

	@echo "🔍 Checking with flake8..."
	poetry run flake8 $(SRC)
	@echo "🧠 Type checking with mypy..."
	poetry run mypy $(SRC)
	@echo "🕵️ Black format check..."
	poetry run black --check $(SRC)
	@echo "🧹 isort check..."
	poetry run isort --check $(SRC)

## Clean cache files
clean:
	@echo "Cleaning cache files..."
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache

test:
	@echo "Running tests..."
	poetry run pytest tests/ -v

## Run tests
tests: test



# Start Docker containers
docker-start:
	docker compose start

# Stop Docker containers
docker-stop:
	docker compose stop

docker-up:
	docker compose up -d

docker-down:
	docker compose down

## Build containers
docker-build:
	docker compose build

## Rebuild Docker containers
docker-rebuild:
	docker compose down -v
	$(MAKE) docker-build
	$(MAKE) docker-up

# Remove project-related images
docker-clean:
	docker compose down --volumes --rmi all



backend:
	@echo "$$(tput bold)Starting backend:$$(tput sgr0)"
	poetry run uvicorn app.main:app --host localhost --reload --port 8000



## Show help
help:
	@echo "$$(tput bold)Available commands:$$(tput sgr0)"
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')