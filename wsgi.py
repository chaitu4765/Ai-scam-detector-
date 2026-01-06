import sys
import os

# Create a clear path to the backend
backend_dir = os.path.join(os.path.dirname(__file__), 'app', 'backend')
sys.path.append(backend_dir)

# Import the 'app' instance from app.backend.app
try:
    from app import app
except ImportError:
    # Fallback for different environment structures
    import app as app_module
    app = app_module.app

if __name__ == "__main__":
    app.run()
