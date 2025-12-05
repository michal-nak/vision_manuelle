"""
UI components creation separated from main app logic
"""
import tkinter as tk
from tkinter import ttk
from ..core.config import COLOR_PALETTE


class UIComponents:
    """Handles creation of UI widgets and layout"""
    
    @staticmethod
    def create_main_layout(root):
        """Create main frame layout"""
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        left_frame = tk.Frame(main_frame, width=400)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_frame.pack_propagate(False)
        
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        return main_frame, left_frame, right_frame
    
    @staticmethod
    def create_camera_panel(parent, mode_var, debug_var, detection_mode, mode_callback, debug_callback):
        """Create camera feed panel with controls"""
        tk.Label(parent, text="Camera Feed", font=("Arial", 12, "bold")).pack()
        
        # Mode selector
        mode_frame = tk.Frame(parent)
        mode_frame.pack(pady=5)
        tk.Label(mode_frame, text="Detection Mode:", font=("Arial", 9)).pack(side=tk.LEFT)
        mode_combo = ttk.Combobox(mode_frame, textvariable=mode_var, 
                                   values=["mediapipe", "cv"], state="readonly", width=12)
        mode_combo.pack(side=tk.LEFT, padx=5)
        mode_combo.bind("<<ComboboxSelected>>", mode_callback)
        
        # Debug toggle
        debug_frame = tk.Frame(parent)
        debug_frame.pack(pady=2)
        tk.Checkbutton(debug_frame, text="Debug Mode", variable=debug_var, 
                      command=debug_callback).pack()
        
        # Camera display
        camera_label = tk.Label(parent, bg="black")
        camera_label.pack(fill=tk.BOTH, expand=True)
        
        return camera_label
    
    @staticmethod
    def create_gesture_info_panel(parent):
        """Create gesture and state information panel"""
        gesture_label = tk.Label(parent, text="Gesture: None", font=("Arial", 10))
        gesture_label.pack(pady=5)
        
        state_label = tk.Label(parent, text="State: IDLE", 
                              bg="gray", fg="white", font=('Arial', 10, 'bold'), width=20)
        state_label.pack(pady=2)
        
        return gesture_label, state_label
    
    @staticmethod
    def create_color_indicator(parent, initial_color):
        """Create color indicator panel"""
        color_frame = tk.Frame(parent)
        color_frame.pack(pady=5)
        tk.Label(color_frame, text="Current Color:", font=("Arial", 9)).pack(side=tk.LEFT)
        color_indicator = tk.Label(color_frame, text="  ", bg=initial_color, 
                                   width=4, relief=tk.SOLID, borderwidth=2)
        color_indicator.pack(side=tk.LEFT, padx=5)
        return color_indicator
    
    @staticmethod
    def create_instructions_panel(parent):
        """Create drawing instructions panel"""
        instructions_frame = tk.LabelFrame(parent, text="Drawing Instructions", 
                                          font=('Arial', 9, 'bold'))
        instructions_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=5)
        
        # CV Mode instructions
        cv_text = """
═══ CV Mode (Finger Count) ═══
• 1 Finger:  Draw with current color
• 2 Fingers: Erase mode
• 3 Fingers: Cycle through colors
• 4 Fingers: Increase brush size
• 5 Fingers: Clear entire canvas
• 0 Fingers: No action (idle)
        """
        tk.Label(instructions_frame, text=cv_text, justify=tk.LEFT, 
                font=("Courier", 8), bg="#f0f0f0").pack(pady=(5,0), padx=5, fill=tk.X)
        
        # MediaPipe mode instructions
        mp_text = """
═══ MediaPipe Mode (Gestures) ═══
• Thumb + Index:  Draw (Pen)
• Thumb + Middle: Erase
• Thumb + Ring:   Cycle colors
• Thumb + Pinky:  Clear canvas
• Index + Middle: Increase brush size
• Middle + Ring:  Decrease brush size
        """
        tk.Label(instructions_frame, text=mp_text, justify=tk.LEFT, 
                font=("Courier", 8), bg="#f0f0f0").pack(pady=(10,5), padx=5, fill=tk.X)
    
    @staticmethod
    def create_toolbar(parent, color_callback, choose_color_callback, 
                      size_change_callback, clear_callback, save_callback, initial_size):
        """Create drawing toolbar with controls"""
        toolbar = tk.Frame(parent, padx=5, pady=5)
        toolbar.pack(side=tk.LEFT, fill=tk.Y)
        
        # Color palette
        colors = ["#000000", "#ffffff", "#FF0000", "#00FF00", "#0000FF", 
                 "#FFFF00", "#FFA500", "#800080"]
        tk.Label(toolbar, text="Colors").pack(pady=(0,5))
        for c in colors:
            b = tk.Button(toolbar, bg=c, width=3, height=1, 
                         command=lambda col=c: color_callback(col))
            b.pack(pady=2)
        
        tk.Button(toolbar, text="Choose Color", 
                 command=choose_color_callback).pack(pady=8, fill=tk.X)
        
        # Brush size control
        tk.Label(toolbar, text="Size").pack(pady=(10,0))
        size_label = tk.Label(toolbar, text=f"{initial_size}")
        size_label.pack()
        size_scale = tk.Scale(toolbar, from_=1, to=50, orient=tk.HORIZONTAL, 
                             command=size_change_callback)
        size_scale.set(initial_size)
        size_scale.pack()
        
        # Action buttons
        tk.Button(toolbar, text="Clear All", 
                 command=clear_callback).pack(pady=12, fill=tk.X)
        tk.Button(toolbar, text="Save", 
                 command=save_callback).pack(pady=2, fill=tk.X)
        
        return size_label, size_scale
    
    @staticmethod
    def create_canvas(parent, bg_color):
        """Create main drawing canvas"""
        canvas = tk.Canvas(parent, bg=bg_color, cursor="cross")
        canvas.pack(fill=tk.BOTH, expand=True)
        return canvas
