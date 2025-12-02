SHELL = bash

# load .env file and export all vars
include .env
export

app := subtitles
# pass --gpus=all only when nvidia-smi available
gpus := $(shell which nvidia-smi > /dev/null && echo "--gpus=all" || echo "")

# if Linux, which normally means the connection is on a remote vm, no browser
launch_browser := $(shell [ "${os}" = "Linux" ] && echo --no-launch-browser)

platform := --platform=linux/amd64

docker_run := docker run -it --rm \
				-v .:/workspace \
				--env-file .env \
				--shm-size=2g \
				${gpus} \
				${platform}

.PHONY: all clean test build shell gradio ckpts config

chown:
	sudo chown -R ${USER}:${USER} .

init: chown
	touch .env

prune:
	docker image prune -f

latest:
	docker images --filter "reference=${app}:latest"

# build with no cache:
# export NO_CACHE=--no-cache
build: prune
	time docker build ${NO_CACHE} \
		${platform} \
		--progress=plain \
		-t ${app}:latest \
		.
	make latest

build-no-cache:
	export NO_CACHE=--no-cache; make build

show-images:
	docker images | grep ^${app}

shell: init
	${docker_run} \
		--entrypoint=/bin/bash \
		${app}:latest

server:
	${docker_run} \
		-e PORT=${PORT} \
		-p ${PORT}:${PORT} \
		--entrypoint=/opt/conda/bin/python \
		${app}:latest \
		server.py

pytest:
	pytest -svx --durations=0 --cache-clear tests/

pre-commit-setup:
	pip install -U pre-commit ruff
	pre-commit install
