"""
Alias for src/explore_datasets.py to maintain project naming compatibility.
"""
import sys
from pathlib import Path

# Ensure src directory is in sys.path for clean imports
sys.path.insert(0, str(Path(__file__).parent))

from explore_datasets import explore_datasets

if __name__ == "__main__":
    target_path = sys.argv[1] if len(sys.argv) > 1 else "datasets"
    explore_datasets(target_path)
