"""
Path utilities for cross-platform support (Windows/Linux).
Handles automatic path detection and resolution.
"""

from pathlib import Path
import os


def get_project_root() -> Path:
    """
    Get the project root directory (works on both Windows and Linux).
    
    Returns:
        Path object pointing to project root
    """
    # Start from this file's location
    current = Path(__file__).resolve()
    
    # Go up to project root (src/utils/paths.py -> src -> project_root)
    project_root = current.parent.parent.parent
    
    return project_root


def resolve_data_root(data_root: str) -> str:
    """
    Resolve data_root path, handling 'auto' keyword.
    
    Args:
        data_root: Either 'auto' or a path string
        
    Returns:
        Resolved absolute path as string
    """
    if data_root == "auto":
        return str(get_project_root())
    else:
        return data_root


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path (string or Path)
        
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_absolute_path(path: str, relative_to: str = None) -> Path:
    """
    Get absolute path, handling both relative and absolute paths.
    
    Args:
        path: Path to resolve
        relative_to: Base path for relative paths (default: project root)
        
    Returns:
        Absolute Path object
    """
    path_obj = Path(path)
    
    if path_obj.is_absolute():
        return path_obj
    else:
        base = Path(relative_to) if relative_to else get_project_root()
        return (base / path_obj).resolve()

