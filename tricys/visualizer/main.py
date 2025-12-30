# tricys/visualizer/main.py
import argparse
import webbrowser
from threading import Timer

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
        "--port", type=int, default=8050, help="Port to run the web server on."
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to bind the web server to."
    )
    args = parser.parse_args()

    app = create_app(h5_file_path=args.h5file)

    url = f"http://{args.host}:{args.port}"

    # Open the web browser in a separate thread to avoid blocking
    Timer(1, lambda: webbrowser.open(url)).start()

    print("Starting Tricys HDF5 Visualizer...")
    print(f"Open your browser and go to: {url}")

    app.run(port=args.port, host=args.host, debug=False)


if __name__ == "__main__":
    start()
