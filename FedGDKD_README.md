# FedGDKD Additional Information

This repository forked from the original FedML repository. However, it has been modified for the project: 
*FedGDKD: Federated GAN-Based Data-Free Knowledge Distillation*. 

Please follow the original README.md for setup instructions.

To run an experiment please take a look at the Makefile for an example, in particular:

NOTE: This will take around 20 hours to run :/.

`make run-example-experiments`

The implementations of algorithms can be found in `./fedml_api/standalone`. To run particular algorithms with more specific
arguments please look at `./fedml_experiments/standalone` for their corresponding experiment class.

## Ablation study branches

For the ablation studies on co-distillation and shared generator please look at the `ablation-co-distillation` branch 
and the `ablation-shared-generator` branch.