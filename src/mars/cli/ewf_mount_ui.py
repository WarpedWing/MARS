#!/usr/bin/env python3
"""
EWF Mount UI - EWF/E01 forensic image mounting interface.

Handles mounting and unmounting of EWF/E01 forensic disk images.
Extracted from app.py to improve modularity.
"""

from __future__ import annotations

import subprocess
import time
from typing import TYPE_CHECKING

from rich.prompt import Prompt

from mars.cli.explorer import browse_for_file

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from rich.console import Console

# Rich Colors
DSB1 = "deep_sky_blue1"
BDSB1 = "bold deep_sky_blue1"


class EWFMountUI:
    """UI for mounting EWF/E01 forensic images."""

    def __init__(self, console: Console):
        """
        Initialize EWF Mount UI.

        Args:
            console: Rich console instance
        """
        self.console = console

    def show_menu(self, show_banner_callback: Callable | None = None) -> None:
        """
        Display EWF mount interface and handle mounting/unmounting.

        Args:
            show_banner_callback: Optional callback to display application banner
        """
        from mars.pipeline.mount_utils.e01_mounter import E01Mounter

        self.console.clear()
        if show_banner_callback:
            show_banner_callback()
        self.console.print(f"\n[{BDSB1}]Mount EWF/E01 Image[/{BDSB1}]\n")

        # Browse for E01 file
        self.console.print("[cyan]Select an E01/EWF forensic image file...[/cyan]\n")
        e01_path = browse_for_file(
            start_path=None,  # Use OS-appropriate default
            file_filter=[".E01", ".e01", ".ex01", ".EX01", ".Ex01"],
            title="Select E01/EWF Image File",
        )

        if not e01_path:
            self.console.print("[yellow]Mount cancelled.[/yellow]")
            time.sleep(0.7)
            return

        self.console.print(f"\n[{DSB1}]Selected:[/{DSB1}] {e01_path}\n")

        # Ask about sudo mounting
        use_sudo = self._prompt_sudo()

        # Create mounter instance
        try:
            mounter = E01Mounter()
        except SystemExit:
            self.console.print("[red]Error: Could not initialize E01 mounter. Check that ewfmount is available.[/red]")
            Prompt.ask("\nPress Enter to continue")
            return

        # Mount the image
        self.console.print("[cyan]Mounting image...[/cyan]\n")
        try:
            raw_mount, fs_mount, disk_device = mounter.mount_e01(e01_path, use_sudo=use_sudo)

            # Show mount information
            self._display_mount_info(raw_mount, fs_mount, disk_device)

            # Wait for unmount command
            self._wait_for_unmount()

            # Unmount everything
            self._unmount_all(raw_mount, fs_mount, disk_device, use_sudo)

            # Invalidate sudo credentials if they were used
            mounter.invalidate_sudo()

            self.console.print("\n[bold dark_sea_green4]Cleanup complete.[/bold dark_sea_green4]\n")

        except Exception as e:
            self.console.print(f"\n[bold red]Error mounting image:[/bold red] {e}\n")
            # Invalidate sudo credentials if they were used
            if "mounter" in locals():
                mounter.invalidate_sudo()

        Prompt.ask("\nPress Enter to continue")

    def _prompt_sudo(self) -> bool:
        """
        Prompt user about sudo mounting and validate credentials.

        Returns:
            True if sudo should be used, False otherwise
        """
        self.console.print("[yellow]Mount with administrator privileges (sudo)?[/yellow]")
        self.console.print("[dim](Required for accessing system-protected files in macOS)[/dim]")
        use_sudo_response = Prompt.ask("\nMount with sudo? (y/n)", choices=["y", "n"], default="n")
        use_sudo = use_sudo_response.lower() == "y"

        # Validate sudo access if requested
        if use_sudo:
            self.console.print("\n[cyan]Requesting administrator privileges...[/cyan]")
            try:
                subprocess.run(["sudo", "-v"], check=True, capture_output=True, timeout=60)
                self.console.print("[green][OK] Administrator privileges granted[/green]\n")
            except subprocess.CalledProcessError:
                self.console.print("[red]X Failed to authenticate[/red]")
                use_sudo = False
            except subprocess.TimeoutExpired:
                self.console.print("[red]X Authentication timed out[/red]")
                use_sudo = False

        return use_sudo

    def _display_mount_info(self, raw_mount: Path, fs_mount: Path | None, disk_device: str) -> None:
        """
        Display mount information to user.

        Args:
            raw_mount: Path to raw EWF mount
            fs_mount: Path to filesystem mount (may be None)
            disk_device: Disk device identifier
        """
        self.console.print(f"\n{'=' * 80}")
        self.console.print("[bold dark_sea_green4]Image Mounted Successfully[/bold dark_sea_green4]")
        self.console.print(f"{'=' * 80}")
        self.console.print(f"[{DSB1}]Raw mount:[/{DSB1}] {raw_mount}")
        self.console.print(f"[{DSB1}]Disk device:[/{DSB1}] {disk_device}")
        if fs_mount:
            self.console.print(f"[{DSB1}]Filesystem mount:[/{DSB1}] {fs_mount}")
        else:
            self.console.print("[yellow]Filesystem: Not mounted[/yellow]")
        self.console.print(f"{'=' * 80}\n")

    def _wait_for_unmount(self) -> None:
        """Wait for user to request unmount."""
        self.console.print("[bold yellow]Image is now mounted and accessible.[/bold yellow]")
        self.console.print("[bold cyan]Enter 'u' to unmount and clean up.[/bold cyan]\n")

        while True:
            cmd = (
                Prompt.ask(
                    "[bold cyan]Command[/bold cyan]",
                    show_default=False,
                )
                .strip()
                .lower()
            )

            if cmd == "u":
                break
            self.console.print("[yellow]Invalid command. Enter 'u' to unmount.[/yellow]")

    def _unmount_all(
        self,
        raw_mount: Path,
        fs_mount: Path | None,
        disk_device: str,
        use_sudo: bool,
    ) -> None:
        """
        Unmount all mounted filesystems and devices.

        Args:
            raw_mount: Path to raw EWF mount
            fs_mount: Path to filesystem mount (may be None)
            disk_device: Disk device identifier
            use_sudo: Whether to use sudo for unmounting
        """
        self.console.print("\n[cyan]Unmounting...[/cyan]\n")

        # Unmount filesystem if it was mounted
        if fs_mount:
            self._unmount_filesystem(fs_mount, use_sudo)

        # Detach disk device
        self._detach_disk(disk_device, use_sudo)

        # Unmount raw EWF mount
        self._unmount_raw(raw_mount, use_sudo)

    def _unmount_filesystem(self, fs_mount: Path, use_sudo: bool) -> None:
        """Unmount filesystem mount point.

        Uses diskutil which handles busy mounts more gracefully than raw umount.
        Falls back to force unmount if normal unmount fails (e.g., Finder has folder open).
        """
        unmounted = False

        # Try diskutil unmount first (handles busy mounts better than raw umount)
        try:
            diskutil_cmd = (
                ["sudo", "diskutil", "unmount", str(fs_mount)] if use_sudo else ["diskutil", "unmount", str(fs_mount)]
            )
            subprocess.run(
                diskutil_cmd,
                check=True,
                capture_output=True,
                timeout=30,
            )
            self.console.print(f"[green][OK][/green] Unmounted filesystem: {fs_mount}")
            unmounted = True
        except subprocess.CalledProcessError:
            # Try force unmount if normal unmount fails (e.g., Finder has folder open)
            try:
                force_cmd = (
                    ["sudo", "diskutil", "unmount", "force", str(fs_mount)]
                    if use_sudo
                    else ["diskutil", "unmount", "force", str(fs_mount)]
                )
                subprocess.run(
                    force_cmd,
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
                self.console.print(f"[green][OK][/green] Unmounted filesystem (force): {fs_mount}")
                unmounted = True
            except Exception as e:
                self.console.print(f"[yellow][i][/yellow] Could not unmount filesystem: {e}")
                self.console.print("[dim]    (Will be unmounted when disk is detached)[/dim]")
        except Exception as e:
            self.console.print(f"[yellow][i][/yellow] Could not unmount filesystem: {e}")
            self.console.print("[dim]    (Will be unmounted when disk is detached)[/dim]")

        # Try to remove mount point directory if unmounted
        if unmounted:
            try:
                fs_mount.rmdir()
                self.console.print(f"[green][OK][/green] Removed mount point: {fs_mount}")
            except OSError:
                self.console.print(f"[yellow][i][/yellow] Could not remove mount point: {fs_mount}")

    def _detach_disk(self, disk_device: str, use_sudo: bool) -> None:
        """Detach disk device."""
        try:
            detach_cmd = ["sudo", "hdiutil", "detach", disk_device] if use_sudo else ["hdiutil", "detach", disk_device]
            subprocess.run(
                detach_cmd,
                check=True,
                capture_output=True,
                timeout=30,
            )
            self.console.print(f"[green][OK][/green] Detached disk device: {disk_device}")
        except Exception as e:
            self.console.print(f"[red][X][/red] Error detaching disk: {e}")

    def _unmount_raw(self, raw_mount: Path, use_sudo: bool) -> None:
        """Unmount raw EWF mount."""
        try:
            raw_umount_cmd = ["sudo", "umount", str(raw_mount)] if use_sudo else ["umount", str(raw_mount)]
            subprocess.run(
                raw_umount_cmd,
                check=True,
                capture_output=True,
                timeout=30,
            )
            self.console.print(f"[green][OK][/green] Unmounted raw EWF mount: {raw_mount}")

            # Try to remove raw mount directory
            try:
                raw_mount.rmdir()
                self.console.print(f"[green][OK][/green] Removed mount point: {raw_mount}")
            except OSError:
                self.console.print(f"[yellow][i][/yellow] Could not remove mount point: {raw_mount}")
        except Exception as e:
            self.console.print(f"[red][X][/red] Error unmounting raw mount: {e}")
