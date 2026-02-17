from pathlib import Path
import sys

def get_repo_root():
    """Locates the project root directory using README.md as a marker."""
    if 'google.colab' in sys.modules:
        return Path('/content')

    # Start from the location of THIS file
    try:
        current = Path(__file__).resolve().parent
        while current != current.parent:
            if (current / "README.md").exists():
                return current
            current = current.parent
    except NameError:
        return Path.cwd()
    
    return Path.cwd()

# Define the root
REPO_ROOT = get_repo_root()

# THIS IS THE MISSING FUNCTION
def get_path(*folders):
    """
    Combines project root with subfolders.
    Usage: get_path("data", "raw") -> project_root/data/raw
    """
    return REPO_ROOT.joinpath(*folders)