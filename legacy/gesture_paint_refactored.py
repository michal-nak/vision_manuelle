"""
Refactored Gesture Paint Application - Main UI
Uses modular components for better maintainability
"""
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from collections import deque

from ..detectors import MediaPipeDetector, CVDetector
from ..core.config import (
    UI_WINDOW_WIDTH, UI_WINDOW_HEIGHT,
    COLOR_PALETTE
)
from ..core.utils import FPSCounter
from .canvas_controller import CanvasController
from .gesture_handler import GestureHandler
from .camera_thread import CameraThread


class GesturePaintApp:
    """Main gesture paint application - refactored"""
    
    def __init__(self, root, detection_mode="mediapipe"):
        self.root = root
        self.root.title("Gesture Paint - Refactored")
        self.root.geometry(f"{UI_WINDOW_WIDTH}x{UI_WINDOW_HEIGHT}")
        
        self.detection_mode = detection_mode
        self.detector = None
        
        # Hand tracking
        self.hand_x = 0.5
        self.hand_y = 0.5
        self.prev_hand_x = 0.5
        self.prev_hand_y = 0.5
        self.smoothing_factor = 0.5
        
        # FPS counter
        self.fps_counter = FPSCounter()
        
        # Initialize components
        self.canvas_controller = None
        self.gesture_handler = GestureHandler()
        self.camera_thread = None
        
        # UI elements
        self.camera_label = None
        self.canvas = None
        self.state_label = None
        self.debug_text = None
        
        # Initialize
        self.init_detector()
        self.create_widgets()
        self.setup_gesture_callbacks()
        self.start_camera()
        self.update_ui()
    
    def init_detector(self):
        """Initialize hand detector"""
        if self.detector:
            try:
                self.detector.cleanup()
            except:
                pass
        
        if self.detection_mode == "mediapipe":
            self.detector = MediaPipeDetector()
        else:
            self.detector = CVDetector()
    
    def create_widgets(self):
        """Create UI layout - simplified"""
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Camera
        left_frame = tk.Frame(main_frame, width=400)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_frame.pack_propagate(False)
        
        tk.Label(left_frame, text="Camera Feed", font=("Arial", 12, "bold")).pack()
        
        # Mode selector
        mode_frame = tk.Frame(left_frame)
        mode_frame.pack(pady=5)
        tk.Label(mode_frame, text="Detection Mode:", font=("Arial", 9)).pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value=self.detection_mode)
        mode_combo = ttk.Combobox(mode_frame, textvariable=self.mode_var,
                                   values=["mediapipe", "cv"], state="readonly", width=12)
        mode_combo.pack(side=tk.LEFT, padx=5)
        mode_combo.bind("<<ComboboxSelected>>",
                       lambda e: self.switch_detection_mode(self.mode_var.get()))
        
        self.camera_label = tk.Label(left_frame, bg="black")
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        self.fps_label = tk.Label(left_frame, text="FPS: 0", font=("Arial", 9))
        self.fps_label.pack()
        
        self.gesture_label = tk.Label(left_frame, text="Gesture: None", font=("Arial", 10))
        self.gesture_label.pack(pady=5)
        
        # State indicator
        self.state_label = tk.Label(left_frame, text="State: IDLE",
                                    bg="gray", fg="white", font=('Arial', 10, 'bold'), width=20)
        self.state_label.pack(pady=2)
        
        # Debug log
        debug_frame = tk.LabelFrame(left_frame, text="Debug Log", font=('Arial', 9, 'bold'))
        debug_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=5)
        
        self.debug_text = tk.Text(debug_frame, height=6, bg="black", fg="lime",
                                  font=('Courier', 7), state=tk.DISABLED, wrap=tk.WORD)
        self.debug_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        debug_scroll = tk.Scrollbar(debug_frame, command=self.debug_text.yview)
        debug_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.debug_text.config(yscrollcommand=debug_scroll.set)
        
        # Right panel - Canvas
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        tk.Label(right_frame, text="Drawing Canvas", font=("Arial", 12, "bold")).pack()
        
        # Canvas (create first so controller can be initialized)
        self.canvas = tk.Canvas(right_frame, bg="white", cursor="crosshair")
        
        # Initialize canvas controller
        self.canvas_controller = CanvasController(self.canvas)
        
        # Controls (need canvas_controller to be initialized)
        controls = self._create_controls(right_frame)
        controls.pack(pady=5)
        
        # Pack canvas after controls
        self.canvas.pack(fill=tk.BOTH, expand=True)
    
    def _create_controls(self, parent):
        """Create control buttons"""
        frame = tk.Frame(parent)
        
        tk.Button(frame, text="Clear", command=self.canvas_controller.clear_canvas,
                 width=10).pack(side=tk.LEFT, padx=2)
        tk.Button(frame, text="Brush", command=self.canvas_controller.use_brush,
                 width=10).pack(side=tk.LEFT, padx=2)
        tk.Button(frame, text="Eraser", command=self.canvas_controller.use_eraser,
                 width=10).pack(side=tk.LEFT, padx=2)
        
        tk.Label(frame, text="Size:").pack(side=tk.LEFT, padx=5)
        size_scale = tk.Scale(frame, from_=1, to=50, orient=tk.HORIZONTAL, length=150,
                             command=lambda v: self.canvas_controller.set_brush_size(int(v)))
        size_scale.set(5)
        size_scale.pack(side=tk.LEFT)
        
        return frame
    
    def setup_gesture_callbacks(self):
        """Setup gesture action callbacks"""
        def on_draw(changed):
            self.canvas_controller.use_brush()
            self.canvas_controller.drawing_enabled = True
            self.state_label.config(text="State: DRAWING", bg="green")
            return "DRAWING"
        
        def on_erase(changed):
            self.canvas_controller.use_eraser()
            self.canvas_controller.drawing_enabled = True
            self.state_label.config(text="State: ERASING", bg="orange")
            return "ERASING"
        
        def on_clear(changed):
            if changed:
                self.canvas_controller.clear_canvas()
                self.state_label.config(text="State: CLEARED", bg="red")
                return "CLEARED"
            return None
        
        def on_none(changed):
            self.canvas_controller.drawing_enabled = False
            self.state_label.config(text="State: IDLE", bg="gray")
            return None
        
        self.gesture_handler.register_callback("Draw", on_draw)
        self.gesture_handler.register_callback("Erase", on_erase)
        self.gesture_handler.register_callback("Clear", on_clear)
        self.gesture_handler.register_callback("None", on_none)
    
    def switch_detection_mode(self, mode):
        """Switch between detection modes"""
        if mode == self.detection_mode:
            return
        self.detection_mode = mode
        
        if self.camera_thread:
            self.camera_thread.stop()
        
        self.init_detector()
        self.start_camera()
    
    def start_camera(self):
        """Start camera thread"""
        self.camera_thread = CameraThread(self.detector)
        if not self.camera_thread.start():
            print("Failed to open camera")
    
    def update_ui(self):
        """Update UI elements"""
        if self.camera_thread and self.camera_thread.is_running():
            result = self.camera_thread.get_latest_result()
            
            if result and result['detected']:
                # Update hand position
                raw_x, raw_y = result['hand_center']
                
                if self.prev_hand_x == 0 and self.prev_hand_y == 0:
                    self.hand_x = raw_x
                    self.hand_y = raw_y
                else:
                    self.hand_x = self.prev_hand_x + self.smoothing_factor * (raw_x - self.prev_hand_x)
                    self.hand_y = self.prev_hand_y + self.smoothing_factor * (raw_y - self.prev_hand_y)
                
                self.prev_hand_x = self.hand_x
                self.prev_hand_y = self.hand_y
                
                # Process gesture
                gesture = result.get('gesture', 'None')
                self.gesture_handler.process_gesture(gesture)
                
                # Update debug log
                try:
                    self.debug_text.config(state=tk.NORMAL)
                    self.debug_text.delete(1.0, tk.END)
                    for log_entry in self.gesture_handler.get_recent_log():
                        self.debug_text.insert(tk.END, log_entry + "\n")
                    self.debug_text.see(tk.END)
                    self.debug_text.config(state=tk.DISABLED)
                except:
                    pass
                
                # Draw on canvas
                self.update_canvas_drawing()
            
            # Update camera feed
            if result and 'annotated_frame' in result:
                frame = result['annotated_frame']
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = img.resize((380, 285))
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)
        
        self.fps_label.configure(text=f"FPS: {self.fps_counter.get_fps()}")
        self.gesture_label.configure(text=f"Gesture: {self.gesture_handler.current_gesture}")
        
        self.root.after(16, self.update_ui)
    
    def update_canvas_drawing(self):
        """Update canvas with hand position"""
        canvas_x = int(self.hand_x * self.canvas.winfo_width())
        canvas_y = int(self.hand_y * self.canvas.winfo_height())
        
        # Draw status overlay
        self.canvas_controller.draw_status_overlay(
            canvas_x, canvas_y, self.detection_mode,
            self.canvas_controller.drawing_enabled
        )
        
        if not self.canvas_controller.drawing_enabled:
            return
        
        # Draw
        if self.canvas_controller.last_x is not None:
            self.canvas_controller.draw_line(
                self.canvas_controller.last_x,
                self.canvas_controller.last_y,
                canvas_x, canvas_y
            )
        else:
            self.canvas_controller.draw_point(canvas_x, canvas_y)
        
        self.canvas_controller.last_x = canvas_x
        self.canvas_controller.last_y = canvas_y
    
    def cleanup(self):
        """Cleanup resources"""
        if self.camera_thread:
            self.camera_thread.stop()
        if self.detector:
            try:
                self.detector.cleanup()
            except:
                pass
