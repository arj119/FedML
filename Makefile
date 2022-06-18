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
	./run_fed_experiment.sh fedgdkd mnist mnist
	./run_fed_experiment.sh feddtg mnist mnist
	./run_fed_experiment.sh fd_faug mnist mnist

run-mnist-experiments-colab:
	./run_fed_experiment_colab.sh baseline mnist mnist
	./run_fed_experiment_colab.sh centralised mnist mnist
	./run_fed_experiment_colab.sh fedavg_multiclient mnist mnist
	./run_fed_experiment_colab.sh fedmd mnist mnist
	./run_fed_experiment_colab.sh fedgdkd mnist mnist
	./run_fed_experiment_colab.sh feddtg mnist mnist
	./run_fed_experiment_colab.sh fd_faug mnist mnist

run-fedgan-experiments:
	./run_fed_experiment.sh fedgan mnist mnist hetero 0.5 4 0.25 50 5 GAN_fid 1 10 10 ./experiment_client_configs/homogeneous_all_participating.json
	./run_fed_experiment.sh fedgan mnist mnist hetero 0.1 4 0.25 50 5 GAN_fid 1 10 10 ./experiment_client_configs/homogeneous_all_participating.json
	./run_fed_experiment.sh fedgan mnist mnist hetero 0.05 3 0.25 50 5 GAN_fid 1 10 10 ./experiment_client_configs/homogeneous_all_participating.json

run-fd_faug-experiments:
	./run_fed_experiment.sh fd_faug mnist mnist hetero 0.1 0 0.1 50 5 all_participate 1 10 10 ./experiment_client_configs/homogeneous_all_participating.json
	./run_fed_experiment.sh fd_faug mnist mnist hetero 0.05 0 0.25 50 5 all_participate 1 10 10 ./experiment_client_configs/homogeneous_all_participating.json
	./run_fed_experiment.sh fd_faug mnist mnist hetero 0.05 0 0.1 50 5 all_participate 1 10 10 ./experiment_client_configs/homogeneous_all_participating.json

run-baseline-5-experiments:
	./run_fed_experiment.sh baseline mnist mnist hetero 0.5 1 0.25 50 5 all_participate 1 10 10 ./experiment_client_configs/homogeneous_all_participating.json
	./run_fed_experiment.sh baseline mnist mnist hetero 0.1 4 0.25 50 5 all_participate 1 10 10 ./experiment_client_configs/homogeneous_all_participating.json
	./run_fed_experiment.sh baseline mnist mnist hetero 0.05 2 0.25 50 5 all_participate 1 10 10 ./experiment_client_configs/homogeneous_all_participating.json
	./run_fed_experiment.sh baseline mnist mnist hetero 0.5 3 0.1 50 5 all_participate 1 10 10 ./experiment_client_configs/homogeneous_all_participating.json
	./run_fed_experiment.sh baseline mnist mnist hetero 0.1 0 0.1 50 5 all_participate 1 10 10 ./experiment_client_configs/homogeneous_all_participating.json
	./run_fed_experiment.sh baseline mnist mnist hetero 0.05 0 0.1 50 5 all_participate 1 10 10 ./experiment_client_configs/homogeneous_all_participating.json
	./run_fed_experiment.sh baseline emnist emnist hetero 0.5 1 0.25 50 5 all_participate 1 10 10 ./experiment_client_configs/homogeneous_all_participating.json
	./run_fed_experiment.sh baseline emnist emnist hetero 0.1 4 0.25 50 5 all_participate 1 10 10 ./experiment_client_configs/homogeneous_all_participating.json
	./run_fed_experiment.sh baseline emnist emnist hetero 0.05 1 0.25 50 5 all_participate 1 10 10 ./experiment_client_configs/homogeneous_all_participating.json
	./run_fed_experiment.sh baseline emnist emnist hetero 0.5 4 0.1 50 5 all_participate 1 10 10 ./experiment_client_configs/homogeneous_all_participating.json
	./run_fed_experiment.sh baseline emnist emnist hetero 0.1 1 0.1 50 5 all_participate 1 10 10 ./experiment_client_configs/homogeneous_all_participating.json
	./run_fed_experiment.sh baseline emnist emnist hetero 0.05 3 0.1 50 5 all_participate 1 10 10 ./experiment_client_configs/homogeneous_all_participating.json

