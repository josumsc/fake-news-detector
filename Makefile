lint:
	docker run --rm -i hadolint/hadolint < Dockerfile

run:
	docker-compose -f docker-compose.yml up -d --remove-orphans --build --force-recreate
	@echo "App deployed at http://localhost:5001"

stop:
	docker-compose -f docker-compose.yml down
