SHELL := /bin/bash 

install:
	./CI-install.sh

run-example-experiments:
	sh ./scripts/experiments/run_fed_experiment.sh baseline mnist mnist hetero 0.1 0 0.1 50 5 all_participate 1 10 10 ./experiment_client_configs/homogeneous_all_participating.json
	sh ./scripts/experiments/run_fed_experiment.sh centralised mnist mnist hetero 0.1 0 0.1 50 5 all_participate 1 10 10 ./experiment_client_configs/homogeneous_all_participating.json
	sh ./scripts/experiments/run_fed_experiment.sh fedavg mnist mnist hetero 0.1 0 0.1 50 5 all_participate 1 10 10 ./experiment_client_configs/homogeneous_all_participating.json
	sh ./scripts/experiments/run_fed_experiment.sh fedmd mnist mnist hetero 0.1 0 0.1 50 5 all_participate 1 10 10 ./experiment_client_configs/homogeneous_all_participating.json
	sh ./scripts/experiments/run_fed_experiment.sh fd_faug mnist mnist hetero 0.1 0 0.1 50 5 all_participate 1 10 10 ./experiment_client_configs/homogeneous_all_participating.json
	sh ./scripts/experiments/run_fed_experiment.sh feddtg mnist mnist hetero 0.1 0 0.1 50 5 all_participate 1 10 10 ./experiment_client_configs/homogeneous_all_participating.json
	sh ./scripts/experiments/run_fed_experiment.sh fedgdkd mnist mnist hetero 0.1 0 0.1 50 5 all_participate 1 10 10 ./experiment_client_configs/homogeneous_all_participating.json

run-presentation-experiment:
	sh ./scripts/experiments/run_fed_experiment.sh fedgdkd mnist mnist hetero 0.5 4 0.25 50 5 presentation 1 10 10 ./experiment_client_configs/homogeneous_all_participating.json