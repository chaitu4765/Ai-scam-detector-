import sys
import os

# Add the backend directory to the search path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_path = os.path.join(current_dir, 'app', 'backend')
if backend_path not in sys.path:
    sys.path.append(backend_path)

# Import the 'app' object from app.py inside backend
try:
    from app import app
except ImportError as e:
    print(f"Error importing app: {e}")
    # Fallback to direct import if needed
    import app as app_module
    app = app_module.app

if __name__ == "__main__":
    app.run()
