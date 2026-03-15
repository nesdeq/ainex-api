#!/usr/bin/env python3
"""
Demo Mirror Runner

Usage:
    python run.py                                    # Local mode
    python run.py --remote-vision 192.168.0.3:9999  # Remote mode
    python run.py --help                             # All options
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from demo_mirror.main import main

if __name__ == "__main__":
    main()
