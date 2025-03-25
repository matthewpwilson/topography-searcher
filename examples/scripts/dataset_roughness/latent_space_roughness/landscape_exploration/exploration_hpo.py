## Global optimisation of the interpolation function

# IMPORTS
import logging
import math
from pathlib import Path
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
from topsearch.transition_states.hybrid_eigenvector_following import HybridEigenvectorFollowing
from topsearch.transition_states.nudged_elastic_band import NudgedElasticBand
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

def explore(trial: optuna.Trial, percent_pairs=100, ts_steps=200):
    ktn = exploration(trial, percent_pairs, ts_steps)
    return ktn.n_ts/len(ktn.pairlist)

def exploration(trial: optuna.Trial, percent_pairs=100, ts_steps=200):
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
    
    # Single ended transition state search object that locates transition
    # states from a single starting position, using hybrid eigenvector-following
    hef = HybridEigenvectorFollowing(potential=interpolator,
                                    ts_conv_crit=trial.suggest_float("ts_conv_crit", low=1e-6, high=1e-1),
                                    ts_steps=ts_steps,
                                    pushoff=trial.suggest_float("pushoff", low=1e-3, high=1),
                                    steepest_descent_conv_crit=trial.suggest_float("steepest_descent_conv_crit", low=1e-6, high=1e-2), # convergence criterion for local minimisation
                                    max_uphill_step_size=trial.suggest_float("max_uphill_step_size", low=0.1, high=10), # largest allowed step in a single direction
                                    min_uphill_step_size=trial.suggest_float("min_uphill_step_size", low=1e-10, high=1e-6), # smallest allowed step in a single direction
                                    eigenvalue_conv_crit=trial.suggest_float("eignevalue_conv_crit", low=1e-6, high=1e-2), # convergence criterion for finding the smallest eigenvalue
                                    positive_eigenvalue_step=trial.suggest_float("eigenvalue_step", low=1e-2, high=1))
    
    # Double ended transition state search that locates approximate minimum energy
    # pathways between two points using the nudged elastic band algorithm
    neb = NudgedElasticBand(potential=interpolator,
                            force_constant=trial.suggest_float("force_constant", low=10.0, high=1000.0),
                            image_density=trial.suggest_float("image_density", low=5, high=50),
                            max_images=trial.suggest_int("max_images", low=10, high=50),
                            neb_conv_crit=trial.suggest_float("neb_conv_crit", low=1e-2, high=1))

    # Object that deals with sampling of configuration space,
    # given the object for global optimisation
    explorer = NetworkSampling(ktn=ktn, coords=coords,
                            global_optimiser=optimiser,
                            single_ended_search=hef,
                            double_ended_search=neb,
                            similarity=comparer,
                            multiprocessing_on=True,
                            n_processes=15)

    step_taking.max_displacement = trial.suggest_float('max_displacement', 0.1, 1.0)
    explorer.get_minima(coords=coords,
                        n_steps=5,
                        conv_crit=trial.suggest_float('conv_crit', 1e-6, 1e-1),
                        temperature=trial.suggest_float('temperature', 10.0, 100.0),
                        test_valid=False,
                        test_valid_lbfgs=True,
                        initial_positions=model_data.training)
    
    if ktn.n_minima > 0: 
        energies = get_minima_energies(ktn)
        trial.set_user_attr("smallest_min_energy", np.where(energies > 0, energies, np.inf).min())
    else:
        energies = [math.inf]

    trial.set_user_attr("n_minima", ktn.n_minima)
    

    ktn.dump_network(f".trial_{trial.number}")

    explorer.get_transition_states(method='ClosestEnumeration',
                                            cycles=8,
                                            remove_bounds_minima=True,
                                            trial=trial,
                                            percent_pairs=percent_pairs,
                                            connection_ratio=True)
    ktn.dump_network(f".trial_{trial.number}")

    return ktn

class StopWhenTrialKeepBeingPrunedCallback:
    def __init__(self, threshold: int=3):
        self.threshold = threshold
        self._consequtive_pruned_count = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.PRUNED:
            self._consequtive_pruned_count += 1
        else:
            self._consequtive_pruned_count = 0

        if self._consequtive_pruned_count >= self.threshold:
            logger.info(f"Stopping optimization due to {self._consequtive_pruned_count} pruned trials")
            study.stop()

class StopWhenTSSearchnGoalReached:
    def __init__(self, goal: float = 0.75, warmup_trials: int = 3):
        self.goal = goal
        self.warmup_trials = warmup_trials

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if len(study.get_trials(False, states=[optuna.trial.TrialState.COMPLETE])) > self.warmup_trials:
            if study.best_value > self.goal:
                logger.info(f"Stopping optimization because best study found {study.best_trial.value} TSs out of {study.best_trial.user_attrs['attempted_pairs']} pairs attempted")
                study.stop()



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

    model_data_all =  ModelData(training_file='../data_generation/expected_output/selfies-ted-mini-training.txt', # position of data points in feature space
                        response_file='../data_generation/expected_output/selfies-ted-mini-response.txt') # corresponding response values
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

    model_data_subset = model_data_all.data_subset(percent_points=1)
    
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

    basin_hopping_steps = model_data_subset.training.shape[0]
    minima_pair_steps = 20
    study = optuna.create_study(study_name=Path(__file__).resolve().parent.name, load_if_exists=True, direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=basin_hopping_steps+minima_pair_steps, n_min_trials=3), storage=storage)
    starting_params_schwefel = {
        "max_displacement": 1.0,
        "conv_crit": 1e-5,
        "temperature": 100.0,
        "ts_conv_crit": 1e-4,
        "pushoff": 1.0,
        "steepest_descent_conv_crit":1e-6,
        "max_uphill_step_size": 1,
        "min_uphill_step_size": 1e-7,
        "eigenvalue_conv_crit": 1e-5,
        "positive_eigenvalue_step": 0.1,
        "force_constant": 10,
        "image_density": 10.0,
        "max_images": 45,
        "neb_conv_crit": 1e-2
    }
    starting_params_dataset = {
        "max_displacement": 1.0,
        "conv_crit": 1e-4,
        "temperature": 100.0,
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
    starting_params_latent_space = {
        "max_displacement": 1.0,
        "conv_crit": 1e-2,
        "temperature": 100.0,
        "ts_conv_crit": 5e-2,
        "pushoff":5e-3,
        "steepest_descent_conv_crit":1e-4,
        "max_uphill_step_size": 3e-1,
        "min_uphill_step_size": 1e-7,
        "eigenvalue_conv_crit": 1e-2,
        "positive_eigenvalue_step": 3e-1,
        "force_constant": 5e2,
        "image_density": 10.0,
        "max_images": 30,
        "neb_conv_crit": 0.1
    }
    study.enqueue_trial(starting_params_latent_space)
    study.enqueue_trial(starting_params_dataset)
    study.enqueue_trial(starting_params_schwefel)
    study.optimize(lambda trial: explore(trial, 100, 10), n_trials=100, n_jobs=1, show_progress_bar=True, callbacks=[StopWhenTSSearchnGoalReached(), StopWhenTrialKeepBeingPrunedCallback(10)])

    logger.info(f"Best connection ratio during optimisation = {study.best_value}")

    logger.info(f"Total trials: {len(study.trials)}, Completed trials: {len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))} Pruned: {len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))} ")
    logger.info(f"Best hyperparameters: {study.best_trial.params}")
    model_data = model_data_all
    interpolator = DatasetRegression(model_data_all, model_type="KNeighbors", neighbors=interpolation_study.best_params["interpolation_neighbours"])

    ktn = exploration(study.best_trial)
    # Dump the minima we found to files min.data and min.coords

    ktn.dump_network()
    # Give the energy and position of the global minimum
    logger.info(f"Total minima = {ktn.n_minima}")
    if ktn.n_minima > 0:
        energies = get_minima_energies(ktn)
        logger.info(f"Global minimum energy = {np.min(energies)}")
        logger.info(f"Smallest positive energy = {np.where(energies > 0, energies, np.inf).min()}")
        logger.info(f"Total transistion states during optimisation = {ktn.n_ts}, pairs: {len(ktn.pairlist)}, ratio: {ktn.n_ts/len(ktn.pairlist)}")
