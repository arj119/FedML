SHELL := /bin/bash 

install:
	./CI-install.sh

run-fedavg-experiments:
	./fedavg_multiclient.sh

run-fedmd-experiments:
	./fedmd.sh

run-federated_arjun-experiments:
	./fedarjun.sh

run-fd_faug-experiments:
	./fd_faug.sh

run-fedssgan-experiments:
	./fedssgan.sh

run-mnist-experiments:
	./run_fed_experiment.sh baseline mnist mnist
	./run_fed_experiment.sh centralised mnist mnist
	./run_fed_experiment.sh fedavg_multiclient mnist mnist
	./run_fed_experiment.sh fedmd mnist mnist
	./run_fed_experiment.sh feddtg_arjun mnist mnist
	./run_fed_experiment.sh feddtg mnist mnist
	./run_fed_experiment.sh fd_faug mnist mnist

run-mnist-experiments-colab:
	./run_fed_experiment_colab.sh baseline mnist mnist
	./run_fed_experiment_colab.sh centralised mnist mnist
	./run_fed_experiment_colab.sh fedavg_multiclient mnist mnist
	./run_fed_experiment_colab.sh fedmd mnist mnist
	./run_fed_experiment_colab.sh feddtg_arjun mnist mnist
	./run_fed_experiment_colab.sh feddtg mnist mnist
	./run_fed_experiment_colab.sh fd_faug mnist mnist