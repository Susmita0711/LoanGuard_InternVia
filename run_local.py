"""run_local.py
-----------------
Utility to dynamically find free ports, start Docker containers for FastAPI and Streamlit,
and manage their lifecycle. It writes a temporary .env file which Docker Compose consumes.
"""
import os
import sys
import subprocess
import socket
import time
from pathlib import Path

# Desired default ports (used if free)
DEFAULT_BACKEND_PORT = 8001
DEFAULT_FRONTEND_PORT = 8502


def is_port_free(port: int) -> bool:
    """Check if a TCP port is available on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def find_free_port(start: int) -> int:
    port = start
    while not is_port_free(port):
        port += 1
    return port


def wait_for_service(url: str, timeout: int = 30) -> bool:
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    backend_port = find_free_port(DEFAULT_BACKEND_PORT)
    frontend_port = find_free_port(DEFAULT_FRONTEND_PORT)

    # Write .env for docker‑compose
    env_path = Path('.env')
    env_path.write_text(
        f"BACKEND_PORT={backend_port}\n"
        f"FRONTEND_PORT={frontend_port}\n"
        f"FRONTEND_ORIGIN=http://localhost:{frontend_port}\n"
    )

    print(f"Launching Docker containers (backend:{backend_port}, frontend:{frontend_port}) …")
    # Build and start containers in detached mode
    subprocess.run(["docker", "compose", "up", "-d", "--build"], check=True)

    # Wait for services to become healthy
    if not wait_for_service(f"http://127.0.0.1:{backend_port}/health"):
        print("Backend failed to start within timeout.")
        subprocess.run(["docker", "compose", "down"], stdout=subprocess.DEVNULL)
        sys.exit(1)
    if not wait_for_service(f"http://127.0.0.1:{frontend_port}"):
        print("Frontend failed to start within timeout.")
        subprocess.run(["docker", "compose", "down"], stdout=subprocess.DEVNULL)
        sys.exit(1)

    # Train the model inside the API container if not already present
    print("Training model inside API container …")
    subprocess.run(["docker", "compose", "exec", "api", "python", "src/train.py"], check=True)
    print("Evaluating model …")
    subprocess.run(["docker", "compose", "exec", "api", "python", "src/evaluate.py"], check=True)

    print("\nLoanGuard successfully started!\n")
    print("\nLoanGuard successfully started!\n")
    print(f"Dashboard: http://localhost:{frontend_port}")
    print(f"API: http://localhost:{backend_port}")
    print(f"API Documentation: http://localhost:{backend_port}/docs\n")
    print("Press CTRL+C to stop the services.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down Docker containers…")
        subprocess.run(["docker", "compose", "down"], stdout=subprocess.DEVNULL)
        print("All services stopped.")

if __name__ == "__main__":
    main()
