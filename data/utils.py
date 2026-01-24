"""
Utilities Module
General utility functions for cleanup and maintenance.
"""

import shutil
import time
from pathlib import Path


def cleanup_temp_files(temp_dir: str, retries: int = 3, delay: float = 0.2) -> None:
    """
    Clean up temporary files and directories with retry mechanism for Windows file locks.

    Args:
        temp_dir: Path to the temporary directory to remove
        retries: Number of retry attempts (default: 3)
        delay: Delay between retries in seconds (default: 0.2)
    """
    if not temp_dir or not Path(temp_dir).exists():
        return

    for attempt in range(retries):
        try:
            shutil.rmtree(temp_dir)
            print(f"Pliki tymczasowe usunięte: {temp_dir}")
            return
        except PermissionError as e:
            if attempt < retries - 1:
                # Wait a bit for Windows to release file locks
                time.sleep(delay)
            else:
                print(f"Ostrzeżenie: Nie można usunąć plików tymczasowych: {temp_dir}")
                print(f"Błąd: {e}")
                print("Możesz usunąć je ręcznie później.")
        except Exception as e:
            print(f"Ostrzeżenie: Błąd przy usuwaniu plików tymczasowych: {e}")
            return
