import os
import platform
import subprocess
import time
from typing import List, Dict, Any, Optional

import psutil

def process_list(
    limit: int = 0,
    include_username: bool = True,
    include_cpu: bool = True,
    include_memory: bool = True,
) -> List[Dict[str, Any]]:
    """List running processes with optional limits."""
    attrs = ["pid", "name", "status"]
    if include_username:
        attrs.append("username")
    if include_cpu:
        attrs.append("cpu_percent")
    if include_memory:
        attrs.append("memory_percent")
    processes: List[Dict[str, Any]] = []
    max_items = max(0, int(limit or 0))
    for proc in psutil.process_iter(attrs):
        try:
            processes.append(proc.info)
            if max_items and len(processes) >= max_items:
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return processes

def process_kill(pid: int) -> bool:
    """Kill a process by PID"""
    proc = psutil.Process(pid)
    proc.kill()
    return True

def get_system_info() -> Dict[str, Any]:
    """Get system information (CPU, Memory, Disk)"""
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        "cpu_percent": cpu_percent,
        "memory": {
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent
        },
        "disk": {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": disk.percent
        }
    }


def process_start(command: Any, cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None,
                  use_shell: bool = False) -> Dict[str, Any]:
    """Start a process and return pid details."""
    if not command:
        return {"success": False, "error": "Missing command"}
    try:
        proc = subprocess.Popen(command, cwd=cwd, env=env, shell=use_shell)
        return {"success": True, "pid": proc.pid, "command": command, "shell": use_shell}
    except Exception as exc:
        return {"success": False, "error": str(exc), "command": command, "shell": use_shell}


def service_list() -> Dict[str, Any]:
    """List system services with status."""
    system_name = platform.system().lower()
    if system_name == "windows":
        cmd = ["sc", "query", "state=", "all"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "platform": "windows",
        }
    cmd = ["systemctl", "list-units", "--type=service", "--all", "--no-pager"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "success": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "platform": system_name,
    }


def service_start(name: str) -> Dict[str, Any]:
    """Start a system service."""
    if not name:
        return {"success": False, "error": "Missing service name"}
    system_name = platform.system().lower()
    if system_name == "windows":
        cmd = ["sc", "start", name]
    else:
        cmd = ["systemctl", "start", name]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "success": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "service": name,
    }


def service_stop(name: str) -> Dict[str, Any]:
    """Stop a system service."""
    if not name:
        return {"success": False, "error": "Missing service name"}
    system_name = platform.system().lower()
    if system_name == "windows":
        cmd = ["sc", "stop", name]
    else:
        cmd = ["systemctl", "stop", name]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "success": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "service": name,
    }


def port_list(kind: str = "inet", status: Optional[str] = "LISTEN") -> List[Dict[str, Any]]:
    """List network connections (default: listening ports)."""
    results: List[Dict[str, Any]] = []
    try:
        conns = psutil.net_connections(kind=kind)
    except Exception:
        conns = []
    for conn in conns:
        if status and conn.status != status:
            continue
        laddr = None
        if conn.laddr:
            laddr = {"ip": conn.laddr.ip, "port": conn.laddr.port}
        raddr = None
        if conn.raddr:
            raddr = {"ip": conn.raddr.ip, "port": conn.raddr.port}
        results.append({
            "fd": conn.fd,
            "family": str(conn.family),
            "type": str(conn.type),
            "local": laddr,
            "remote": raddr,
            "status": conn.status,
            "pid": conn.pid,
        })
    return results


def resource_sample() -> Dict[str, Any]:
    """Sample CPU, memory, disk, and network usage."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(os.path.abspath(os.sep))
    net = psutil.net_io_counters()
    now = time.time()
    return {
        "ts": now,
        "cpu_percent": cpu_percent,
        "memory": {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent,
        },
        "disk": {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": disk.percent,
        },
        "network": {
            "bytes_sent": net.bytes_sent,
            "bytes_recv": net.bytes_recv,
            "packets_sent": net.packets_sent,
            "packets_recv": net.packets_recv,
        },
    }
