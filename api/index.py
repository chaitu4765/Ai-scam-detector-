import sys
import os

# Add the project root and app/backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../app/backend'))

from app.backend.app import app
