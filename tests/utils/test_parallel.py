from multiprocessing import Manager, current_process
from assertpy import assert_that
from topsearch.utils.parallel import run_parallel


def test_calls_function_with_each_arg():
    results = run_parallel(to_upper, ["dog", "cat"])
    assert_that(list(results)).contains("DOG", "CAT")

def test_returns_inputs():
    results = run_parallel(to_upper, ["dog", "cat"], return_input=True)
    assert_that(list(results)).contains_only(("dog","DOG"), ("cat", "CAT"))
 
def test_runs_in_parallel():
    with Manager() as manager:
        blocker = manager.Event()
        results = run_parallel(to_upper_blocking, ["first", "second"], [blocker])
        assert_that(next(results)).is_equal_to("SECOND")
        blocker.set()
        assert_that(next(results)).is_equal_to("FIRST") 

def test_resuses_process_pool():
    results1 = list(run_parallel(return_process_id, ["dog", "cat"], processes=2))
    results2 = list(run_parallel(return_process_id, ["dog", "cat"], processes=2))
    assert_that(results2).contains_only(*results1)

def test_creates_new_process_pool_when_requested_size_different():
    results1 = list(run_parallel(return_process_id, ["dog", "cat", "mouse"], processes=2))
    results2 = list(run_parallel(return_process_id, ["dog", "cat", "mouse"], processes=3))
    assert_that(results2).does_not_contain(*results1)
    from topsearch.utils.parallel import executor
    assert_that(executor._max_workers).is_equal_to(3)

def return_process_id(input: str):
    return current_process().name

def to_upper_blocking(data: str, blocker) -> str:
        if data == "first":
            blocker.wait(1)

        return data.upper()

def to_upper(data: str) -> str:
    return data.upper()


