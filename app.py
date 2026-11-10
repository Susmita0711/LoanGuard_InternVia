"""app.py
-----------------
Entry point to start the whole application using run_local.
"""
import subprocess
import sys

if __name__ == "__main__":
    # Delegates to run_local which handles port detection and process management
    subprocess.run([sys.executable, "run_local.py"], check=True)

