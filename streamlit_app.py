"""Root entrypoint for Streamlit Cloud deployment."""
import sys
from pathlib import Path

# Ensure repository root is on sys.path
root = str(Path(__file__).resolve().parent)
if root not in sys.path:
    sys.path.insert(0, root)

from app.streamlit_app import main

if __name__ == "__main__":
    main()
