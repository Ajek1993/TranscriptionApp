"""
Output Manager Module
Centralized output management for user-facing messages.
"""

from tqdm import tqdm


class OutputManager:
    """Centralized output management for user-facing messages."""

    @staticmethod
    def stage_header(stage_num: int, stage_name: str) -> None:
        """Print stage header: === Etap X: NAME ==="""
        print(f"\n=== Etap {stage_num}: {stage_name} ===")

    @staticmethod
    def info(message: str, use_tqdm_safe: bool = False) -> None:
        """Print info message (use tqdm.write if progress bars active)."""
        if use_tqdm_safe:
            tqdm.write(message)
        else:
            print(message)

    @staticmethod
    def success(message: str) -> None:
        """Print success message with checkmark."""
        print(f"\n[OK] {message}")

    @staticmethod
    def warning(message: str, use_tqdm_safe: bool = False) -> None:
        """Print warning message."""
        msg = f"Ostrzeżenie: {message}"
        if use_tqdm_safe:
            tqdm.write(msg)
        else:
            print(msg)

    @staticmethod
    def error(message: str) -> None:
        """Print error message."""
        print(f"Błąd: {message}")

    @staticmethod
    def detail(message: str, use_tqdm_safe: bool = False) -> None:
        """Print detailed info (indented, secondary importance)."""
        msg = f"  {message}"
        if use_tqdm_safe:
            tqdm.write(msg)
        else:
            print(msg)

    @staticmethod
    def mode_header(mode_name: str, details: dict = None) -> None:
        """Print mode header with configuration details."""
        print(f"\n=== {mode_name} ===")
        if details:
            for key, value in details.items():
                print(f"{key}: {value}")
