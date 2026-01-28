import logging
import os
from typing import Optional, Union

logger = logging.getLogger(__name__)


def get_safe_max_workers(
    user_requested_limit: Optional[Union[int, str]] = None, maximize: bool = False
) -> int:
    """
    Determines the safe number of workers based on turbo mode, user request, and defaults.

    Priority:
    1. maximize=True (Turbo Mode) -> Use all available cores (os.cpu_count()).
    2. user_requested_limit -> Use the requested number (as long as it's >= 1).
    3. Default -> Use 50% of available cores (os.cpu_count() // 2).

    Args:
        user_requested_limit: The number of workers requested by the user config.
        maximize: If True, ignore safety limits and use all cores.

    Returns:
        The actual number of workers to use (integer >= 1).
    """
    total_cores = os.cpu_count() or 1

    # 1. Turbo Mode
    if maximize:
        logger.info(f"Turbo Mode ENABLED: Using all {total_cores} available cores.")
        return total_cores

    # 2. User Config Override
    if user_requested_limit is not None:
        try:
            workers = int(user_requested_limit)
            if workers < 1:
                workers = 1
            logger.info(
                f"Using user-requested max_workers: {workers} (Overrides default safety limit)."
            )
            return workers
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid user_requested_limit: '{user_requested_limit}'. Falling back to default."
            )

    # 3. Default Safe Limit (50%)
    safe_workers = max(1, total_cores // 2)
    logger.info(
        f"Using default safety limit: {safe_workers} workers (50% of {total_cores} cores)."
    )
    return safe_workers
