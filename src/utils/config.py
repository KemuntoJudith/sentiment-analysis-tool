from pathlib import Path
import sys

# Locate the project root directory using README.md as a marker.
def get_repo_root():

    # --- Google Colab ---
    if 'google.colab' in sys.modules:
        print("Running in Google Colab.")
        return Path('/content')

    # --- Local Environment ---
    print("Running locally.")

    # Start from current working directory
    current = Path().resolve()

    # Walk upward until README.md is found
    while current != current.parent:
        if (current / "README.md").exists():
            return current
        current = current.parent

    # If not found, raise error
    raise RuntimeError(
        "Project root not found. Ensure README.md exists in the repository root."
    )


# Define repository root once
REPO_ROOT = get_repo_root()

# Build a path relative to the project root.
def get_path(*folders):
    return REPO_ROOT.joinpath(*folders)

# Create directory if it does not exist and return the path
def ensure_path(*folders):
    
    path = get_path(*folders)
    path.mkdir(parents=True, exist_ok=True)
    return path
