## Overview

This folder contains the core code for Neur2SP.  The design was intended to be relatively modular so that new two-stage stochastic programming
instances can be implemented.  We provide a brief overview of each of the folders/classes:
- `two_sp`: The implementation of solving the extensive form, sampling scenarios, solving second-stage problems, and evaluating first-stage decisions.
- `dm`: The data manager provides the implementation for instance creation and data generation for NN-{E, P}.  
- `models`: The model implementations for LR, NN-E, and NN-P as well as the network architecture classes. These files are agnostic to the 2SP problem.
- `model2mip`: The embedding implementation for LR, NN-E, and NN-P. These files are agnostic to the 2SP problem.
- `approximator`: The surrogate optimization model creation and solving.
- `utils`: General utility files for each problem.  Mostly used for getting paths.
- `scripts`: Contains the set of scripts to run all experiments. These files are agnostic to the 2SP problem.
  - `run_dm.py`: Script to generate instances and datasets.
  - `train_model.py`: Script to train models.
  - `get_best_model.py`: Script to get the best model.  Renames best model to standard name for reading in later aspects.
  - `evaluate_model.py `: Script to solve and evaluate surrogate optimization model.
  - `evaluate_extensive.py`: Evaluates baselines (extensive form).
  - `collect_results.py`: Collects results for a single problem and stores them in a dictionary.
  - `collect_all_results.py`: Collects results for all problems and stores them in a dictionary.
  - `create_random_search.py`(Appendix Only): Generates configs to be run for random search.
  - `get_rs_results.py`(Appendix Only): Collects variance results across random search instances. 
