"""Four-condition inference/export entry point, preserving the tracked runner."""
import sys
from pathlib import Path

# The inspected existing runner imports sibling tools by their bare names.
TOOLS=Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
from run_one_scene_comparison import main

if __name__=='__main__':main()
