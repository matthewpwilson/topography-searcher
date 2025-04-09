from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Collection, List
import multiprocessing as mp
import threadpoolctl
import logging
import os
import math
logger = logging.getLogger("parallel")
logger.debug(threadpoolctl.threadpool_info())
logger.debug(f"cpu_count: {os.cpu_count()}")
executor = None

def run_parallel(func: Callable, arglist: Collection, extra_args: List= [], processes: int = 2, return_input = False, max_threads = os.cpu_count()):
    global executor
    processes = min(processes, len(arglist))
    threads_per_process = int(math.floor(max_threads/processes))
    current_threads = threadpoolctl.threadpool_info()[0]["num_threads"]
    if executor is None or executor._max_workers != processes or current_threads != threads_per_process:
        logger.debug(f"Resizing process pool to {processes} with {threads_per_process} threads per process")
        threadpoolctl.threadpool_limits(threads_per_process)
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
        