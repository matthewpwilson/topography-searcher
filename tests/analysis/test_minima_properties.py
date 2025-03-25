from multiprocessing import current_process
from assertpy import assert_that
import pytest
import numpy as np
import os
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.data.coordinates import StandardCoordinates
from topsearch.similarity.similarity import StandardSimilarity
from topsearch.potentials.test_functions import Camelback, Schwefel
from topsearch.analysis.minima_properties import get_bounds_minima, \
        get_minima_above_cutoff, get_minima_energies, get_ordered_minima, \
        get_all_bounds_minima, get_similar_minima, get_invalid_minima, \
        get_distance_matrix, get_distance_from_minimum, validate_minima
import logging
logging.basicConfig(format='%(asctime)s %(name)-16s %(levelname)-8s %(message)s',
                    level="DEBUG",
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=f"/Users/mattwil/git/TopographySearcher/test_logs/logfile_{current_process().name}")

current_dir = os.path.dirname(os.path.dirname((os.path.realpath(__file__))))

def test_get_invalid_minima():
    coords = StandardCoordinates(ndim=3, bounds=[(-5.0, 5.0),
                                                 (-5.0, 5.0),
                                                 (-5.0, 5.0)])
    schwefel = Schwefel()
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.analysis')
    for i in range(ktn.n_minima):
        minimum = ktn.get_minimum_coords(i)
        ktn.G.nodes[i]['coords'] = minimum*100.0
    minima = get_invalid_minima(ktn, schwefel, coords)
    assert np.all(minima == np.array([]))
    ktn.add_minimum(np.array([6.2541, 113.8487, 426.3035]), -0.89487)
    minima = get_invalid_minima(ktn, schwefel, coords)
    assert np.all(minima == np.array([9]))

def test_validate_minima_with_lbfgs_parallel():
    
    
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    ktn = KineticTransitionNetwork()
    ktn.add_minimum(np.array([0.08984199, -0.7126564 ]), -1.0316284534898734)
    ktn.add_minimum(np.array([0, 1]), 1)
    ktn.add_minimum(np.array([1.60710479, 0.56865138]), 2.104250310311282)
    ktn.add_minimum(np.array([0, 1]), -0)
    ktn.add_minimum(np.array([1.5, 1.5]), -0)

    potential = Camelback()
    invalid_minima = validate_minima(ktn, coords, potential, processes=2)
    assert_that(invalid_minima).contains_only(1, 3, 4)
    
def test_get_bounds_minima():
    coords = StandardCoordinates(ndim=3, bounds=[(-5.0, 5.0),
                                                 (-5.0, 5.0),
                                                 (-5.0, 5.0)])
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.analysis')
    minima = get_bounds_minima(ktn, coords)
    assert minima == [0, 6, 7]

def test_get_all_bounds_minima():
    coords = StandardCoordinates(ndim=3, bounds=[(-5.0, 5.0),
                                                 (-5.0, 5.0),
                                                 (-5.0, 5.0)])
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.analysis')
    ktn.add_minimum(np.array([5.0, 5.0, 5.0]), 0.0)
    minima = get_all_bounds_minima(ktn, coords)
    assert minima == [9]

def test_get_similar_minima():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.analysis')
    points = np.array([[5.0, -3.3082, 4.245]])
    similar_minima = get_similar_minima(ktn, 0.1, points)
    assert similar_minima == [0]
    points = np.array([[5.0, 5.0, 5.0],
                       [5.0, -3.3082, 4.245],
                       [-1.3963, -5.0, -2.062]])
    similar_minima = get_similar_minima(ktn, 0.1, points)
    assert similar_minima == [0, 6]

def test_get_minima_above_cutoff():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.analysis')
    minima = get_minima_above_cutoff(ktn, -2.0)
    assert np.all(minima == np.array([2, 3, 4, 5, 6, 7, 8]))
    minima = get_minima_above_cutoff(ktn, 0.0)
    assert np.all(minima == np.array([]))

def test_get_minima_energies():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.analysis')
    energies = get_minima_energies(ktn)
    assert np.all(energies == pytest.approx(np.array([-2.29238, -2.58207,
                                                      -1.67184, -1.44769,
                                                      -1.35213, -1.71037,
                                                      -1.28440, -1.21125,
                                                      -1.27458])))

def test_get_ordered_minima():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.analysis')
    ordered_indices = get_ordered_minima(ktn)
    assert np.all(ordered_indices == np.array([1, 0, 5, 2,
                                               3, 4, 6, 8, 7]))

def test_get_distance_matrix():
    coords = StandardCoordinates(ndim=3, bounds=[(-5.0, 5.0),
                                                 (-5.0, 5.0),
                                                 (-5.0, 5.0)])
    similarity = StandardSimilarity(0.1, 0.1)
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.analysis')
    ktn.remove_minima(np.array([4, 5, 6, 7, 8]))
    dist_matrix = get_distance_matrix(ktn, similarity, coords)
    assert np.all(dist_matrix[0,:] == pytest.approx([0.0, 6.64842106,
                                                     8.1389305, 7.79272806]))
    assert np.all(dist_matrix[1,:] == pytest.approx([6.64842106, 0.0,
                                                     7.7137482, 9.39845094]))              
    assert np.all(dist_matrix[2,:] == pytest.approx([8.1389305, 7.7137482,
                                                     0.0,  6.61532193]))
    assert np.all(dist_matrix[3,:] == pytest.approx([7.79272806, 9.39845094,
                                                     6.61532193, 0.0]))

def test_get_distance_matrix():
    coords = StandardCoordinates(ndim=3, bounds=[(-5.0, 5.0),
                                                 (-5.0, 5.0),
                                                 (-5.0, 5.0)])
    similarity = StandardSimilarity(0.1, 0.1)
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.analysis')
    ktn.remove_minima(np.array([4, 5, 6, 7, 8]))
    dist_vector = get_distance_from_minimum(ktn, similarity, coords, 0)
    assert np.all(dist_vector == pytest.approx([0.0, 6.64842106,
                                                8.1389305, 7.79272806]))

@pytest.fixture
def ktn_single_minimum() -> KineticTransitionNetwork:
    ktn = KineticTransitionNetwork()
    ktn.add_minimum(np.array([0,0,0]), 0)
    return ktn

@pytest.fixture
def ktn_multiple_minima(ktn_single_minimum) -> KineticTransitionNetwork:
    for i in range(10):
        ktn_single_minimum.add_minimum(np.array([i/10,i/10,i/10]), 0.5)

    return ktn_single_minimum

@pytest.fixture()
def coords_3d() -> StandardCoordinates:
    return StandardCoordinates(ndim=3, bounds=[(-1.0, 1.0),
                                                 (-1.0, 1.0),
                                                 (-1.0, 1.0)])
    
@pytest.mark.parametrize(
    "min_coords",
    [
        ([0,0,0]),
        ([0.0001, 0.0001, 0.0001]),
        ([-0.0003, -0.0003, -0.0003]),
        ([0, 0, 0.000001])
    ]
)
def test_validate_succeeds_when_lbfgs_results_close_to_min_coords(mocker, ktn_single_minimum: KineticTransitionNetwork, coords_3d, min_coords):
    mock_fmin = mocker.patch("topsearch.analysis.minima_properties.fmin_l_bfgs_b")
    mock_fmin.return_value = (np.array(min_coords), 0, None)
    validate_minima(ktn_single_minimum, coords_3d, mocker.MagicMock())

@pytest.mark.parametrize(
    "min_coords",
    [
        ([1,1,1]),
        ([0.01, 0.01, 0.01]),
        ([-0.01, -0.01, -0.01]),
        ([0, 0, 0.01])
    ]
)
def test_validate_throws_when_lbfgs_results_do_not_match_min_coords(mocker, ktn_single_minimum: KineticTransitionNetwork, coords_3d, min_coords):
    mock_fmin = mocker.patch("topsearch.analysis.minima_properties.fmin_l_bfgs_b")
    mock_fmin.return_value = (np.array(min_coords), 0, None)
    
    invalid_minima =  validate_minima(ktn_single_minimum, coords_3d, mocker.MagicMock())
    assert_that(invalid_minima).contains_only(0)

def test_validate_flags_invalid_when_lbfgs_results_do_not_match_min_energy(mocker, ktn_single_minimum: KineticTransitionNetwork, coords_3d):
    mock_fmin = mocker.patch("topsearch.analysis.minima_properties.fmin_l_bfgs_b")
    mock_fmin.return_value = (np.array([0,0,0]), 0.01, None)
    
    assert_that(validate_minima(ktn_single_minimum, coords_3d, mocker.MagicMock())).contains_only(0)

def test_validate_flags_invalid_when_some_lbfgs_results_do_not_match_min_energy(mocker, ktn_multiple_minima: KineticTransitionNetwork, coords_3d):
    mock_fmin = mocker.patch("topsearch.analysis.minima_properties.fmin_l_bfgs_b")
    mock_fmin.return_value = (np.array([0,0,0]), 0.0001, None)
    
    assert_that(validate_minima(ktn_multiple_minima, coords_3d, mocker.MagicMock())).contains_only(*range(1,11))    


def test_validate_succeeds_when_lbfgs_results_close_to_min_energy(mocker, ktn_single_minimum: KineticTransitionNetwork, coords_3d):
    mock_fmin = mocker.patch("topsearch.analysis.minima_properties.fmin_l_bfgs_b")
    mock_fmin.return_value = (np.array([0,0,0]), 0.0001, None)
    invalid_minima = validate_minima(ktn_single_minimum, coords_3d, mocker.MagicMock())      
    assert_that(invalid_minima).is_empty()

def test_validate_succeeds_when_all_lbfgs_results_close_to_min_energy(mocker, ktn_single_minimum: KineticTransitionNetwork, coords_3d):
    ktn_single_minimum.add_minimum(np.array([0.5,0.5,0.5]), 0.5)
    ktn_single_minimum.add_minimum(np.array([1,1,1]), 1)

    mock_fmin = mocker.patch("topsearch.analysis.minima_properties.fmin_l_bfgs_b")
    mock_fmin.side_effect = [(np.array([0,0,0]), 0.0001, None),
                             (np.array([0.5,0.5,0.5]), 0.5001, None),
                             (np.array([1,1,1]), 0.9999, None)]
    invalid_minima = validate_minima(ktn_single_minimum, coords_3d, mocker.MagicMock())      
    assert_that(invalid_minima).is_empty()

def test_validate_succeeds_when_gradient_close_to_zero(mocker, ktn_single_minimum, coords_3d):
    mock_fmin = mocker.patch("topsearch.analysis.minima_properties.fmin_l_bfgs_b")
    mock_fmin.return_value = (np.array([0,0,0]), 0.0001, None)

    interpolation = mocker.MagicMock()
    interpolation.gradient = mocker.MagicMock(return_value=[0.001,0.001,0.001])
    invalid_minima = validate_minima(ktn_single_minimum, coords_3d, interpolation)     
    assert_that(invalid_minima).is_empty()


def test_validate_succeeds_when_gradient_close_to_zero_except_at_upper_bounds(mocker, ktn_single_minimum, coords_3d):
    mock_fmin = mocker.patch("topsearch.analysis.minima_properties.fmin_l_bfgs_b")
    mock_fmin.return_value = (np.array([1,0,1]), 0.0001, None)

    ktn = KineticTransitionNetwork()
    ktn.add_minimum(np.array([1,0,1]), 0)

    interpolation = mocker.MagicMock()
    interpolation.gradient = mocker.MagicMock(return_value=[1,0.001,1])
    invalid_minima = validate_minima(ktn, coords_3d, interpolation) 
    assert_that(invalid_minima).is_empty()


def test_validate_succeeds_when_gradient_close_to_zero_except_at_lower_bounds(mocker, ktn_single_minimum, coords_3d):
    mock_fmin = mocker.patch("topsearch.analysis.minima_properties.fmin_l_bfgs_b")
    mock_fmin.return_value = (np.array([-1,0,-1]), 0.0001, None)

    ktn = KineticTransitionNetwork()
    ktn.add_minimum(np.array([-1,0,-1]), 0)

    interpolation = mocker.MagicMock()
    interpolation.gradient = mocker.MagicMock(return_value=[1,0.001,1])
    invalid_minima = validate_minima(ktn, coords_3d, interpolation)     
    assert_that(invalid_minima).is_empty()

def test_validate_flags_when_gradient_not_close_to_zero(mocker, ktn_single_minimum, coords_3d):
    mock_fmin = mocker.patch("topsearch.analysis.minima_properties.fmin_l_bfgs_b")
    mock_fmin.return_value = (np.array([0,0,0]), 0.0001, None)

    interpolation = mocker.MagicMock()
    interpolation.gradient = mocker.MagicMock(return_value=np.array([1,1,1]))
   
    invalid_minima = validate_minima(ktn_single_minimum, coords_3d, interpolation)   
    assert_that(invalid_minima).contains_only(0)

def test_validate_flags_when_f_plus_less_than_minimum_energy(mocker, ktn_single_minimum, coords_3d):
    mock_fmin = mocker.patch("topsearch.analysis.minima_properties.fmin_l_bfgs_b")
    mock_fmin.return_value = (np.array([0,0,0]), 0.0001, None)

    interpolation = mocker.MagicMock()
    interpolation.gradient = mocker.MagicMock(return_value=np.array([0,0,0]))
    interpolation.function = mocker.MagicMock(return_value=0)
    invalid_minima = validate_minima(ktn_single_minimum, coords_3d, interpolation)          
    assert_that(invalid_minima).contains_only(0)
    


def test_validate_flags_when_f_minus_less_than_minimum_energy(mocker, ktn_single_minimum, coords_3d):
    mock_fmin = mocker.patch("topsearch.analysis.minima_properties.fmin_l_bfgs_b")
    mock_fmin.return_value = (np.array([0,0,0]), 0.0001, None)

    interpolation = mocker.MagicMock()
    interpolation.gradient = mocker.MagicMock(return_value=np.array([0,0,0]))
    interpolation.function = mocker.MagicMock(side_effect=[1,0])
    invalid_minima = validate_minima(ktn_single_minimum, coords_3d, interpolation)               
    assert_that(invalid_minima).contains_only(0)
