DOCKER_USERNAME = josumsc

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black src/*.py

lint:
	pylint --disable=R,C,W1203,E1101 src/.*
	docker run --rm -i hadolint/hadolint < Dockerfile

publish:
	python src/cli.py publish
	docker build -t $(DOCKER_USERNAME)/flask-fake-news:latest .
	docker push $(DOCKER_USERNAME)/flask-fake-news:latest

run:
	docker-compose -f docker-compose.yml up -d --remove-orphans --build --force-recreate
	@echo "App deployed at http://localhost:5001"

stop:
	docker-compose -f docker-compose.yml down
