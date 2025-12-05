"""
Test script for refactored modules
Run this to verify the restructuring works correctly
"""
import tkinter as tk
from src.ui.gesture_paint_refactored import GesturePaintApp


def main():
    """Test the refactored application"""
    root = tk.Tk()
    
    # Test with CV mode
    app = GesturePaintApp(root, detection_mode="cv")
    
    def on_closing():
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
