# formatters/__init__.py

'''
import os
import sys
# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
'''

from formatters.narrative import NarrativeFormatter 
from formatters.structured import StructuredFormatter 
from formatters.json_formatter import JSONFormatter


def get_formatter(name: str):
    if name == "narrative":
        return NarrativeFormatter()
    elif name == "structured":
        return StructuredFormatter()
    elif name == "json":
        return JSONFormatter()
    else:
        raise ValueError(f"Unknown formatter: {name}")
    