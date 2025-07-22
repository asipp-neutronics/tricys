import tkinter as tk
import sys
from .param_config.simulation_ui import SimulationUI

def main():
    """Main function to initialize and run the GUI for parameter configuration."""
    root = tk.Tk()
    app = SimulationUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: [root.destroy(), sys.exit(0)])
    root.mainloop()

if __name__ == "__main__":
    main()