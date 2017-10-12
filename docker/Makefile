help:
	@cat Makefile

DATA?="/mnt/experiments/data"
GPU?=0
DOCKER_FILE=Dockerfile
DOCKER=GPU=$(GPU) nvidia-docker
BACKEND=tensorflow
TEST=tests/
SRC=$(shell dirname `pwd`)

build:
	docker build -t research --build-arg python_version=3.5 -f $(DOCKER_FILE) .

bash: build
	$(DOCKER) run -it -v $(SRC):/src -v $(DATA):/data --env KERAS_BACKEND=$(BACKEND) research bash

ipython: build
	$(DOCKER) run -it -v $(SRC):/src -v $(DATA):/data --env KERAS_BACKEND=$(BACKEND) research ipython

notebook: build
	$(DOCKER) run -it -v $(SRC):/src -v $(DATA):/data -p 8888:8888 -p 6006:6006 --env PASSWORD=open --env KERAS_BACKEND=$(BACKEND) research

test: build
	$(DOCKER) run -it -v $(SRC):/src -v $(DATA):/data --env KERAS_BACKEND=$(BACKEND) research py.test $(TEST)


