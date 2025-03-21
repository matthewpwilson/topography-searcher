## Global optimisation of the interpolation function

# IMPORTS
import logging
import math
import numpy as np
import optuna
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate, KFold

from topsearch.data.coordinates import StandardCoordinates
from topsearch.data.model_data import ModelData
from topsearch.potentials.dataset_fitting import DatasetInterpolation, DatasetRegression
from topsearch.similarity.similarity import StandardSimilarity
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.global_optimisation.perturbations import StandardPerturbation
from topsearch.global_optimisation.basin_hopping import BasinHopping
from topsearch.sampling.exploration import NetworkSampling
from topsearch.analysis.minima_properties import get_minima_energies


## BEGIN CALCULATIONS

import os
os.environ["LOGLEVEL"] = "DEBUG"
from topsearch.utils.logging import configure_logging
configure_logging()
logger = logging.getLogger()

def interpolation(trial):
    interpolation = KNeighborsRegressor(trial.suggest_int("interpolation_neighbours", low=3, high=20))
    cv = KFold(n_splits=5, shuffle=True)
    cv_results = cross_validate(interpolation,
                                model_data_all.training,
                                model_data_all.response,
                                cv=cv,
                                scoring="neg_mean_squared_error",
                                n_jobs=None)
    return cv_results['test_score'].mean() * -1

def minimise(trial):
    return basin_hopping(trial)[0]

def basin_hopping(trial: optuna.Trial):
    ktn = KineticTransitionNetwork()
     # Specify the coordinates for optimisation. We will optimise in a space
# of three dimensions, and select the standard bounds used for interpolation
    coords = StandardCoordinates(ndim=model_data.n_dims, bounds=bounds)
    logger.debug(f"DatasetRegression with {model_data.n_points} points")

     # Global optimisation class using basin-hopping
    optimiser = BasinHopping(ktn=ktn,
                            potential=interpolator,
                            similarity=comparer,
                            step_taking=step_taking)
    # Object that deals with sampling of configuration space,
    # given the object for global optimisation
    explorer = NetworkSampling(ktn=ktn, coords=coords,
                            global_optimiser=optimiser,
                            single_ended_search=None,
                            double_ended_search=None,
                            similarity=comparer,
                            n_processes=14)

    step_taking.max_displacement = trial.suggest_float('max_displacement', 0.1, 1.0)
    explorer.get_minima(coords=coords,
                        n_steps=5,
                        conv_crit=trial.suggest_float('conv_crit', 1e-6, 1e-1),
                        temperature=trial.suggest_float('temperature', 10.0, 100.0),
                        test_valid=False, # Validation is very expensive with KNN in high dimensions
                        initial_positions=model_data.training,
                        trial=trial)
    if ktn.n_minima > 0: 
        energies = get_minima_energies(ktn)
    else:
        energies = [math.inf]

    trial.set_user_attr("n_minima", ktn.n_minima)
    # Optimise for smallest positive minimum energy
    ktn.dump_network(f".trial_{trial.number}")
    return np.where(energies > 0, energies, np.inf).min(), ktn

# study_name = "example-study"  # Unique identifier of the study.
# file_path = "./optuna_journal_storage.log"
# storage = optuna.storages.JournalStorage(
#     optuna.storages.journal.JournalFileBackend(file_path),  # NFS path for distributed optimization
# )
if __name__ == '__main__':    
    # INITIALISATION

    file_path = "./optuna_journal_storage.log"
    storage = optuna.storages.JournalStorage(
               optuna.storages.journal.JournalFileBackend(file_path),  # NFS path for distributed optimization
    )

    model_data_all = ModelData(training_file='/Users/mattwil/git/TopographySearcher/examples/scripts/adme_dataset_optuna/adme_fps_ds_train.txt', # position of data points in feature space
                        response_file='/Users/mattwil/git/TopographySearcher/examples/scripts/adme_dataset_optuna/adme_sol_response_train.txt') # corresponding response values
    bounds = [(0.0, 1.0) for _ in range(model_data_all.n_dims)]
    # Remove duplicate training data
    model_data_all.remove_duplicates()
    # Normalise response and training to lie within the range (0, 1)
    model_data_all.normalise_training()
    model_data_all.normalise_response()

    # Find best shape param for interpolation function on whole dataset
    interpolation_study = optuna.create_study(storage=storage, direction="minimize")
    interpolation_study.optimize(interpolation, n_trials=50, n_jobs=1, show_progress_bar=True)

    logger.info(f"Best interpolation parameters: {interpolation_study.best_params} MSE: {interpolation_study.best_value}")

    model_data_subset = model_data_all.data_subset(percent_points=5)
    
    model_data = model_data_subset
    interpolator = DatasetRegression(model_data_subset, model_type="KNeighbors", neighbors=interpolation_study.best_params["interpolation_neighbours"])

    # Specify the simple test function we will perform global optimisation on
    # Similarity object, decides if two points are the same or different
    # Same if distance between points is less than distance_criterion
    # and the difference in function value is less than energy_criterion
    # Distance is a proportion of the range, in this case 0.05*1000.0
    comparer = StandardSimilarity(distance_criterion=0.05,
                                energy_criterion=1e-2,
                                proportional_distance=True)

    # Perturbation scheme for proposing new positions in configuration space
    # Standard perturbation just applies random perturbations
    step_taking = StandardPerturbation(max_displacement=1.0,
                                    proportional_distance=True)

    # Start with a large conv-crit for quick convergence then increase
    starting_params = {
        "interpolation_neighbours": 20,
        "max_displacement": 1.0,
        "conv_crit": 1e-2,
        "temperature": 100.0
    }
    min_starting_points = np.floor(max(25, model_data_subset.training.shape[0]/4))
    study = optuna.create_study(storage=storage, direction="minimize", pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=min_starting_points))
    study.enqueue_trial(starting_params)
    starting_params.update({"conv_crit":1e-3})
    study.enqueue_trial(starting_params)
    starting_params.update({"conv_crit":1e-4})
    study.enqueue_trial(starting_params)
    study.optimize(minimise, n_trials=50, n_jobs=1, show_progress_bar=True)
    trial_with_smallest_minimum = study.best_trial #min(study.best_trials, key=lambda t: t.values[1])

    logger.info(f"Total trials: {len(study.trials)}, Completed trials: {len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))} Pruned: {len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))} ")
    logger.info(f"Best hyperparameters: {trial_with_smallest_minimum.params}")
    model_data = model_data_all
    interpolator = DatasetRegression(model_data_all, model_type="KNeighbors", neighbors=interpolation_study.best_params["interpolation_neighbours"])

    energy, ktn = basin_hopping(trial_with_smallest_minimum)
    # Dump the minima we found to files min.data and min.coords

    ktn.dump_network(text_string=".min_only")
    # Give the energy and position of the global minimum
    logger.info(f"Total minima = {ktn.n_minima}")
    if ktn.n_minima > 0:
        energies = get_minima_energies(ktn)
        logger.info(f"Global minimum energy = {np.min(energies)}")
        logger.info(f"Smallest positive energy = {np.where(energies > 0, energies, np.inf).min()}")
