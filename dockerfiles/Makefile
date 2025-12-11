.PHONY: clean docker-reset open-cpu

clean:

# For some reason this is sometimes necessary
docker-reset:
	systemctl --user daemon-reload
	systemctl --user restart docker

build-cpu: Dockerfile.cpu
	docker build -t kmeans-cpu -f Dockerfile.cpu ..

open-cpu:
	docker run -it --rm -v ..:/home kmeans-cpu /bin/bash
