SHELL := /bin/bash 

install:
	./CI-install.sh

run-fedavg-experiments:
	./CI-script-fedavg.sh

run-fedmd-experiments:
	./fedmd.sh

run-federated_arjun-experiments:
	./fedarjun.sh