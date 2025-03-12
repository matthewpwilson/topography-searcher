## Generate complete landscape for the two-dimensional Schwefel function
## on both a single processor, and with 6 processors to highlight the
## acceleration possible with multiprocessing 

# IMPORTS

import numpy as np
import optuna
from timeit import default_timer as timer
from topsearch.data.coordinates import StandardCoordinates
from topsearch.data.model_data import ModelData
from topsearch.potentials.dataset_fitting import DatasetInterpolation
from topsearch.potentials.test_functions import Schwefel
from topsearch.similarity.similarity import StandardSimilarity
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.global_optimisation.perturbations import StandardPerturbation
from topsearch.global_optimisation.basin_hopping import BasinHopping
from topsearch.sampling.exploration import NetworkSampling
from topsearch.transition_states.hybrid_eigenvector_following import HybridEigenvectorFollowing
from topsearch.transition_states.nudged_elastic_band import NudgedElasticBand

import os
import logging

os.environ["LOGLEVEL"] = "DEBUG"
from topsearch.utils.logging import configure_logging
configure_logging()

    
def objective(trial):
    return ts_search(trial).n_ts

def ts_search(trial: optuna.trial.Trial):
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_string='.min_only')
    coords = StandardCoordinates(ndim=model_data.n_dims, bounds=bounds)
    # Global optimisation class using basin-hopping
    optimiser = BasinHopping(ktn=ktn,
                            potential=interpolation,
                            similarity=comparer,
                            step_taking=step_taking)
    # Single ended transition state search object that locates transition
    # states from a single starting position, using hybrid eigenvector-following
    hef = HybridEigenvectorFollowing(potential=interpolation,
                                    ts_conv_crit=trial.suggest_float("ts_conv_crit", low=1e-6, high=1e-1),
                                    ts_steps=200,
                                    pushoff=trial.suggest_float("pushoff", low=1e-3, high=1),
                                    steepest_descent_conv_crit=trial.suggest_float("steepest_descent_conv_crit", low=1e-6, high=1e-2), # convergence criterion for local minimisation
                                    max_uphill_step_size=trial.suggest_float("max_uphill_step_size", low=0.1, high=10), # largest allowed step in a single direction
                                    min_uphill_step_size=trial.suggest_float("min_uphill_step_size", low=1e-10, high=1e-6), # smallest allowed step in a single direction
                                    eigenvalue_conv_crit=trial.suggest_float("eignevalue_conv_crit", low=1e-6, high=1e-2), # convergence criterion for finding the smallest eigenvalue
                                    positive_eigenvalue_step=trial.suggest_float("eigenvalue_step", low=1e-2, high=1))
    
    # Double ended transition state search that locates approximate minimum energy
    # pathways between two points using the nudged elastic band algorithm
    neb = NudgedElasticBand(potential=interpolation,
                            force_constant=trial.suggest_float("force_constant", low=10.0, high=1000.0),
                            image_density=trial.suggest_float("image_density", low=5, high=50),
                            max_images=trial.suggest_int("max_images", low=10, high=50),
                            neb_conv_crit=trial.suggest_float("neb_conv_crit", low=1e-2, high=1))

    # Parallel equivalent for searching configuration space
    explorer_parallel = NetworkSampling(ktn=ktn,
                                        coords=coords,
                                        global_optimiser=optimiser,
                                        single_ended_search=hef,
                                        double_ended_search=neb,
                                        similarity=comparer,
                                        multiprocessing_on=True,
                                        n_processes=14)

        # BEGIN CALCULATIONS

    explorer_parallel.get_transition_states(method='ClosestEnumeration',
                                            cycles=8,
                                            remove_bounds_minima=True,
                                            trial=trial)
    
    return ktn

if __name__ == '__main__':
    # INITIALISATION
    logger = logging.getLogger()
    # Specify the coordinates for optimisation. We will optimise in a space
    # of two dimensions, and select the standard bounds used for Schwefel
    # Specify the test function we will generate the landscape of
    model_data = ModelData(training_file='/Users/mattwil/git/TopographySearcher/examples/notebooks/adme_sol_feature_subset_training.txt', # position of data points in feature space
                        response_file='/Users/mattwil/git/TopographySearcher/examples/notebooks/adme_sol_response.txt') # corresponding response values
    bounds = [(0.0, 1.0) for _ in range(model_data.n_dims)]
    # Remove duplicate training data
    model_data.remove_duplicates()
    # Normalise response and training to lie within the range (0, 1)
    model_data.normalise_training()
    model_data.normalise_response()

    interpolation = DatasetInterpolation(model_data=model_data, # dataset to interpolate
                                        smoothness=0.006125153115603497, # tightness with which each data point
                                        shape=6740.289309314874,
                                        kernel="gaussian") 
    # Similarity object, decides if two points are the same or different
    # Same if distance between points is less than distance_criterion
    # and the difference in function value is less than energy_criterion
    # Distance is a proportion of the range, in this case 0.05*1000.0
    comparer = StandardSimilarity(distance_criterion=0.01,
                                energy_criterion=1e-2,
                                proportional_distance=True)
    # Perturbation scheme for proposing new positions in configuration space
    # Standard perturbation just applies random perturbations
    step_taking = StandardPerturbation(max_displacement=1.0,
                                    proportional_distance=True)
    
   
    #  BEGIN CALCULATIONS

    parallel_start_time = timer()
    
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=50, n_min_trials=1))
    starting_params = {
        "ts_conv_crit": 5e-4,
        "pushoff":5e-3,
        "steepest_descent_conv_crit":1e-4,
        "max_uphill_step_size": 1e1,
        "min_uphill_step_size": 1e-8,
        "eigenvalue_conv_crit": 1e-3,
        "positive_eigenvalue_step": 1e-2,
        "force_constant": 5e2,
        "image_density": 50.0,
        "max_images": 30,
        "neb_conv_crit": 1e-2
    }
    study.enqueue_trial(starting_params)
    study.optimize(objective, n_trials=200, n_jobs=1)

    logger.info(f"Total trials: {len(study.trials)}, Completed trials: {len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))} Pruned: {len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))} ")
    logger.info(f"Best hyperparameters: {study.best_trial.params}")

    ktn = ts_search(study.best_trial)


    ktn.dump_network()
   
    print("Total transistion states = ", ktn.n_ts)
    
    parallel_end_time = timer()

    parallel_length = parallel_end_time - parallel_start_time

    print("Total runtime = ", parallel_length)