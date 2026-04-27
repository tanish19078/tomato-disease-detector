import sys
import os

# Add root project dir to path so it can import master_api
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from master_api import app
