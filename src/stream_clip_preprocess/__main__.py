"""Allow running stream-clip-preprocess as a module.

Usage: python -m stream_clip_preprocess
"""

import sys

from stream_clip_preprocess.cli import main

if __name__ == "__main__":
    sys.exit(main())
