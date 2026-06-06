#!/usr/bin/env python
"""Wrapper that just runs scalesim/scale.py with the same args."""
import os, sys, runpy
sys.argv[0] = os.path.join(os.path.dirname(__file__), "scalesim", "scale.py")
runpy.run_path(sys.argv[0], run_name="__main__")
