import socket
from contextlib import closing


def find_free_port(
    start_port: int, host: str = "127.0.0.1", max_retries: int = 100
) -> int:
    """
    Finds a free port starting from a given port.

    Args:
        start_port: The port number to start checking from.
        host: The host to check. Defaults to "127.0.0.1".
        max_retries: The maximum number of ports to check. Defaults to 100.

    Returns:
        A free port number.

    Raises:
        IOError: If no free port is found within the range.
    """
    for port in range(start_port, start_port + max_retries):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            try:
                # Bind specifically to the requested host
                sock.bind((host, port))
                # If bind succeeds, the port is free.
                # We close the socket immediately (via context manager) and return the port.
                return port
            except OSError:
                # Port is in use, check the next one
                continue

    raise IOError(
        f"Could not find a free port on {host} starting from {start_port} within {max_retries} attempts."
    )
