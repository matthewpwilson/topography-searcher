from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Collection, List
import multiprocessing as mp
import threadpoolctl
import logging
import os
import math
from topsearch.utils.logging import configure_logging

logger = logging.getLogger("parallel")
logger.debug(threadpoolctl.threadpool_info())
logger.debug(f"cpu_count: {os.cpu_count()}")
executor = None

def run_parallel(func: Callable, arglist: Collection, extra_args: List= [], processes: int = 2, return_input = False, max_threads: int = 0):
    if len(arglist) == 0:
        logger.debug("Empty list of args")
        return []
    
    global executor
    if max_threads == 0:
        max_threads = processes    
    processes = min(processes, len(arglist))
    threads_per_process = int(math.floor(max_threads/processes))
    current_threads = threadpoolctl.threadpool_info()[0]["num_threads"]
    if executor is None or executor._max_workers != processes or current_threads != threads_per_process:
        logger.debug(f"Resizing process pool to {processes} with {threads_per_process} threads per process")
        threadpoolctl.threadpool_limits(threads_per_process)
        logger.debug(threadpoolctl.threadpool_info())
        executor = ProcessPoolExecutor(processes, mp.get_context("spawn"), pool_worker_init, tuple([threads_per_process]))

    futures = {}
    for arg in arglist:
        future = executor.submit(func, arg, *extra_args)
        futures[future] = arg 

    for future in as_completed(futures):
        if return_input:
            yield futures[future], future.result()
        else:     
            yield future.result()
    
def pool_worker_init(threads: int):
    threadpoolctl.threadpool_limits(threads)
    configure_logging()
    logger = logging.getLogger("parallel")
    logger.debug(f"Worker initializing with {threads} threads cpu_count: {os.cpu_count()}")
    logger.debug(threadpoolctl.threadpool_info())