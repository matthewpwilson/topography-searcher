from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Collection, List
import multiprocessing as mp

executor = None

def run_parallel(func: Callable, arglist: Collection, extra_args: List= [], processes: int = 2, return_input = False):
    global executor
    if executor is None or executor._max_workers != processes:
        executor = ProcessPoolExecutor(processes, mp.get_context("spawn"))

    futures = {}
    for arg in arglist:
        future = executor.submit(func, arg, *extra_args)
        futures[future] = arg 

    for future in as_completed(futures):
        if return_input:
            yield futures[future], future.result()
        else:     
            yield future.result()
        