#!/usr/bin/env python3
"""
Advanced Dexter System Launcher
A comprehensive launcher for the Dexter AI system with advanced features.
"""

import os
import sys
import time
import subprocess
import psutil
import json
import signal
import atexit
from pathlib import Path
from typing import List, Dict, Optional
import threading
import socket


class DexterLauncher:
    def __init__(self):
        self.repo_root = Path(__file__).parent
        self.config = self.load_config()
        self.processes: Dict[str, subprocess.Popen] = {}
        self.log_dir = self.repo_root / "logs"
        self.log_dir.mkdir(exist_ok=True)

        # Register cleanup on exit
        atexit.register(self.cleanup)

    def load_config(self) -> Dict:
        """Load launcher configuration."""
        default_config = {
            "ports": {
                "api_server": 8000,
                "dexter_core": 8001,
                "browser": 3000,
                "browser_debug": 9222,
            },
            "components": {
                "api_server": {
                    "enabled": True,
                    "command": ["python", "core/api.py"],
                    "wait_time": 3,
                },
                "dexter_core": {
                    "enabled": True,
                    "command": ["python", "dexter.py"],
                    "wait_time": 5,
                },
                "browser": {
                    "enabled": True,
                    "command": ["npm", "start"],
                    "working_dir": "dexter-browser",
                    "wait_time": 10,
                },
                "memory_system": {
                    "enabled": True,
                    "command": ["python", "-c", "print('Memory system initialized')"],
                    "wait_time": 2,
                },
            },
            "max_retries": 3,
            "startup_timeout": 30,
        }

        config_file = self.repo_root / "launcher_config.json"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")
                print("Using default configuration")

        return default_config

    def check_port(self, port: int) -> bool:
        """Check if a port is in use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            result = sock.connect_ex(("localhost", port))
            return result == 0

    def kill_process_on_port(self, port: int) -> bool:
        """Kill process running on specified port."""
        try:
            # Find process using the port
            for conn in psutil.net_connections():
                if conn.laddr.port == port and conn.pid:
                    try:
                        process = psutil.Process(conn.pid)
                        print(
                            f"Killing process {process.name()} (PID: {conn.pid}) on port {port}"
                        )
                        process.terminate()
                        process.wait(timeout=5)
                        return True
                    except (
                        psutil.NoSuchProcess,
                        psutil.AccessDenied,
                        psutil.TimeoutExpired,
                    ):
                        pass
            return False
        except Exception as e:
            print(f"Error killing process on port {port}: {e}")
            return False

    def start_component(self, name: str, config: Dict) -> bool:
        """Start a single component."""
        if not config.get("enabled", True):
            print(f"Skipping {name} (disabled in config)")
            return True

        port = self.config["ports"].get(name.replace("_", "_"))
        if port and self.check_port(port):
            print(f"Port {port} is in use, attempting to free it...")
            if not self.kill_process_on_port(port):
                print(f"Warning: Could not free port {port}")
                return False

            # Wait a bit for the port to be freed
            time.sleep(2)

        command = config["command"]
        working_dir = config.get("working_dir", self.repo_root)
        wait_time = config.get("wait_time", 5)

        log_file = self.log_dir / f"{name}.log"

        print(f"Starting {name}...")
        print(f"Command: {' '.join(command)}")
        print(f"Working directory: {working_dir}")

        try:
            with open(log_file, "w") as f:
                process = subprocess.Popen(
                    command,
                    cwd=working_dir,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid if os.name != "nt" else None,
                )

            self.processes[name] = process
            print(f"{name} started with PID: {process.pid}")

            # Wait for component to initialize
            time.sleep(wait_time)

            # Check if process is still running
            if process.poll() is not None:
                print(f"Error: {name} process terminated unexpectedly")
                return False

            print(f"✓ {name} started successfully")
            return True

        except Exception as e:
            print(f"Error starting {name}: {e}")
            return False

    def start_system(self) -> bool:
        """Start the entire Dexter system."""
        print("=" * 50)
        print("  DEXTER ADVANCED SYSTEM LAUNCHER")
        print("=" * 50)
        print()

        # Check prerequisites
        if not self.check_prerequisites():
            return False

        # Clear ports
        print("Checking and clearing ports...")
        for component, config in self.config["components"].items():
            port_key = component.replace("_", "_")
            port = self.config["ports"].get(port_key)
            if port:
                self.kill_process_on_port(port)

        print("Port check complete.")
        print()

        # Start components
        print("Starting Dexter components...")
        print()

        success_count = 0
        total_components = len(
            [c for c in self.config["components"].values() if c.get("enabled", True)]
        )

        for component, config in self.config["components"].items():
            if self.start_component(component, config):
                success_count += 1
            else:
                print(f"Failed to start {component}")

            print()  # Blank line for readability

        print("=" * 50)
        print("  STARTUP COMPLETE")
        print("=" * 50)
        print(f"Successfully started {success_count}/{total_components} components")
        print()
        print("Active components:")
        for name, process in self.processes.items():
            if process.poll() is None:
                print(f"  ✓ {name} (PID: {process.pid})")
            else:
                print(f"  ✗ {name} (terminated)")

        print()
        print("Log files are available in:", self.log_dir)
        print()
        print("To stop all components, run: python advanced_launcher.py --stop")

        return success_count > 0

    def stop_system(self) -> None:
        """Stop all Dexter components."""
        print("Stopping Dexter system...")

        for name, process in self.processes.items():
            if process.poll() is None:
                try:
                    if os.name == "nt":
                        process.terminate()
                    else:
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)

                    print(f"Stopping {name}...")
                    process.wait(timeout=10)
                    print(f"✓ {name} stopped")
                except subprocess.TimeoutExpired:
                    print(f"Force killing {name}...")
                    process.kill()
                except Exception as e:
                    print(f"Error stopping {name}: {e}")

        # Also try to kill any remaining processes
        self.kill_remaining_processes()
        print("System stopped.")

    def kill_remaining_processes(self) -> None:
        """Kill any remaining Dexter-related processes."""
        try:
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    cmdline = " ".join(proc.info["cmdline"] or [])
                    if any(
                        keyword in cmdline
                        for keyword in ["dexter.py", "http.server", "dexter-browser"]
                    ):
                        proc.terminate()
                        proc.wait(timeout=5)
                        print(
                            f"Killed remaining process: {proc.info['name']} (PID: {proc.info['pid']})"
                        )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            print(f"Error killing remaining processes: {e}")

    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met."""
        print("Checking prerequisites...")

        # Check Python
        if (
            not subprocess.run(["python", "--version"], capture_output=True).returncode
            == 0
        ):
            print("Error: Python is not available")
            return False

        # Check if main files exist
        if not (self.repo_root / "dexter.py").exists():
            print("Error: dexter.py not found")
            return False

        # Check Node.js for browser component
        browser_config = self.config["components"].get("browser", {})
        if browser_config.get("enabled", True):
            if (
                not subprocess.run(
                    ["node", "--version"], capture_output=True
                ).returncode
                == 0
            ):
                print("Warning: Node.js not found - browser component may not work")

        print("Prerequisites check complete.")
        return True

    def status(self) -> None:
        """Show system status."""
        print("=" * 50)
        print("  DEXTER SYSTEM STATUS")
        print("=" * 50)
        print()

        # Check ports
        print("Port Status:")
        for component, port in self.config["ports"].items():
            if self.check_port(port):
                print(f"  Port {port} ({component}): IN USE")
            else:
                print(f"  Port {port} ({component}): FREE")
        print()

        # Check processes
        print("Process Status:")
        for name, process in self.processes.items():
            if process.poll() is None:
                print(f"  {name}: RUNNING (PID: {process.pid})")
            else:
                print(f"  {name}: STOPPED")
        print()

        # Check for any Dexter-related processes
        print("All Dexter-related processes:")
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = " ".join(proc.info["cmdline"] or [])
                if any(keyword in cmdline for keyword in ["dexter", "http.server"]):
                    print(
                        f"  {proc.info['name']} (PID: {proc.info['pid']}): {cmdline[:100]}..."
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        print()
        print("Log files:")
        for log_file in self.log_dir.glob("*.log"):
            size = log_file.stat().st_size
            print(f"  {log_file.name}: {size} bytes")

    def cleanup(self) -> None:
        """Cleanup function called on exit."""
        if self.processes:
            print("Cleaning up processes...")
            self.stop_system()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Dexter Advanced System Launcher")
    parser.add_argument("--start", action="store_true", help="Start the system")
    parser.add_argument("--stop", action="store_true", help="Stop the system")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--config", type=str, help="Path to config file")

    args = parser.parse_args()

    launcher = DexterLauncher()

    if args.config:
        # Load custom config
        if Path(args.config).exists():
            with open(args.config) as f:
                launcher.config.update(json.load(f))
        else:
            print(f"Config file not found: {args.config}")
            return 1

    try:
        if args.stop:
            launcher.stop_system()
        elif args.status:
            launcher.status()
        else:
            launcher.start_system()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        launcher.cleanup()
    except Exception as e:
        print(f"Error: {e}")
        launcher.cleanup()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
