'''A collection of useful decorators.'''
import time
import logging
import functools

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def record_time(func):
    """A decorator to print the time it takes for a function to execute."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logger.info(f"Finished '{func.__name__}' in {run_time:.4f} secs")
        return result
    return wrapper
