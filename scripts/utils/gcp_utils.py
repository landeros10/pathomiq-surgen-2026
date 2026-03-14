"""GCP SSH/SCP utilities using sshpass.

Provides a thin subprocess wrapper around sshpass for running commands and
transferring files to/from the GCP server.
"""

import subprocess
from pathlib import Path


def ssh_cmd(
    host: str,
    user: str,
    password: str,
    cmd: str,
    timeout: int = 300,
) -> str:
    """Run *cmd* on the remote host via SSH and return stdout.

    Raises:
        RuntimeError  if the remote command exits with non-zero status.
        subprocess.TimeoutExpired  if the command exceeds *timeout* seconds.
    """
    result = subprocess.run(
        [
            "sshpass", "-p", password,
            "ssh", "-o", "StrictHostKeyChecking=no",
            f"{user}@{host}",
            cmd,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"SSH command failed (exit {result.returncode}):\n"
            f"  cmd : {cmd}\n"
            f"  stderr: {result.stderr.strip()}"
        )
    return result.stdout


def scp_to(
    host: str,
    user: str,
    password: str,
    local_path: "str | Path",
    remote_path: str,
) -> None:
    """Copy *local_path* to *remote_path* on the remote host."""
    subprocess.run(
        [
            "sshpass", "-p", password,
            "scp", "-o", "StrictHostKeyChecking=no",
            str(local_path),
            f"{user}@{host}:{remote_path}",
        ],
        check=True,
    )


def scp_from(
    host: str,
    user: str,
    password: str,
    remote_path: str,
    local_path: "str | Path",
) -> None:
    """Copy *remote_path* from the remote host to *local_path*."""
    subprocess.run(
        [
            "sshpass", "-p", password,
            "scp", "-r", "-o", "StrictHostKeyChecking=no",
            f"{user}@{host}:{remote_path}",
            str(local_path),
        ],
        check=True,
    )
