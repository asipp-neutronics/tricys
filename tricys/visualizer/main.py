# tricys/visualizer/main.py
import argparse
import webbrowser
from threading import Timer

from tricys.utils.network_utils import find_free_port

from .app import create_app


def start():
    """
    Parses command-line arguments and starts the Dash server.
    """
    parser = argparse.ArgumentParser(description="Launch the Tricys HDF5 Visualizer.")
    parser.add_argument(
        "h5file", type=str, nargs="?", help="Path to the HDF5 results file."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to run the web server on. Defaults to first available starting from 8050.",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to bind the web server to."
    )
    args = parser.parse_args()

    if args.port is None:
        try:
            port = find_free_port(8050, host=args.host)
        except IOError as e:
            print(f"Error: {e}")
            return
    else:
        port = args.port

    app = create_app(h5_file_path=args.h5file)

    url = f"http://{args.host}:{port}"

    # Open the web browser in a separate thread to avoid blocking
    Timer(1, lambda: webbrowser.open(url)).start()

    print("Starting Tricys HDF5 Visualizer...")
    print(f"Open your browser and go to: {url}")

    app.run(port=port, host=args.host, debug=False)


if __name__ == "__main__":
    start()
