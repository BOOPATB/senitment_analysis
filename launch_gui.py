#!/usr/bin/env python3
"""
Launch script for the Sentiment Analysis GUI
Starts the Flask server and opens the browser automatically
"""

import sys
import os
import time
import webbrowser
import threading
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from gui_server import app
    import flask
    print("✅ Flask imported successfully")
except ImportError as e:
    print(f"❌ Error importing Flask: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        ('flask', 'flask'),
        ('flask_cors', 'flask_cors'), 
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('scikit-learn', 'sklearn'),
        ('transformers', 'transformers'),
        ('torch', 'torch'),
        ('nltk', 'nltk')
    ]
    
    missing = []
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        print("❌ Missing required packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\nPlease install with: pip install -r requirements.txt")
        return False
    
    return True

def open_browser(url, delay=2):
    """Open browser after a delay"""
    def _open():
        time.sleep(delay)
        print(f"🌐 Opening browser: {url}")
        webbrowser.open(url)
    
    thread = threading.Thread(target=_open, daemon=True)
    thread.start()

def main():
    """Main function to launch the GUI"""
    print("=" * 60)
    print("🧠 SENTIMENT ANALYSIS GUI 3D")
    print("   Powered by Three.js & Anime.js")
    print("=" * 60)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        return
    
    print("✅ All dependencies found")
    
    # Set up environment
    os.environ['FLASK_ENV'] = 'development'
    
    # Configuration
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = True
    
    url = f"http://localhost:{PORT}"
    
    print(f"🚀 Starting server...")
    print(f"📱 Local: {url}")
    print(f"🌐 Network: http://{HOST}:{PORT}")
    print()
    print("💡 Features:")
    print("   • 3D particle visualization with Three.js")
    print("   • Smooth animations with Anime.js")
    print("   • Real-time sentiment analysis")
    print("   • Traditional ML & Transformer models")
    print("   • Batch processing support")
    print("   • Interactive controls & settings")
    print()
    print("⚡ Tips:")
    print("   • Use Ctrl+Enter to analyze text")
    print("   • Click example chips to try samples")
    print("   • Adjust 3D settings in the gear menu")
    print("   • Upload files for batch analysis")
    print()
    print("🔄 Loading models in background...")
    print("   (This may take a minute on first run)")
    print()
    
    # Open browser automatically
    open_browser(url, delay=3)
    
    try:
        # Run the Flask app
        app.run(
            host=HOST,
            port=PORT,
            debug=DEBUG,
            use_reloader=False,  # Disable reloader to prevent double startup
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down server...")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print("Try running on a different port or check if port 5000 is already in use")

if __name__ == "__main__":
    main()
