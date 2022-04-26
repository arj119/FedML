SHELL := /bin/bash 

install:
	./CI-install.sh

run-fedavg-experiments:
	./fedavg-run.sh

run-fedmd-experiments:
	./fedmd.sh

run-federated_arjun-experiments:
	./fedarjun.sh

run-fd_faug-experiments:
	./fd_faug.sh