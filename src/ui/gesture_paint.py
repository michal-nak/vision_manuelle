import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import time
from collections import deque
from ..detectors import MediaPipeDetector, CVDetector
from ..core.config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    POSITION_SMOOTHING, PROCESSING_SCALE,
    UI_WINDOW_WIDTH, UI_WINDOW_HEIGHT,
    COLOR_PALETTE, GESTURE_NAMES
)
from ..core.utils import find_camera, setup_camera, FPSCounter, draw_text_with_background

# Import modular components for code reuse
from .canvas_controller import CanvasController
from .gesture_handler import GestureHandler
from .camera_thread import CameraThread

try:
    import ctypes
    user32 = ctypes.windll.user32
    WINDOWS_API_AVAILABLE = True
except:
    WINDOWS_API_AVAILABLE = False

class GesturePaintApp:
    def __init__(self, root, detection_mode="mediapipe"):
        self.root = root
        root.title("Gesture Paint - Hand Control")
        root.geometry(f"{UI_WINDOW_WIDTH}x{UI_WINDOW_HEIGHT}")
        root.minsize(800, 400)
        
        self.detection_mode = detection_mode
        self.detector = None

        self.brush_color = COLOR_PALETTE[0]
        self.bg_color = "white"
        self.brush_size = 5
        self.eraser_on = False
        self.last_x = None
        self.last_y = None
        self.drawing_enabled = False
        
        self.color_palette = COLOR_PALETTE
        self.current_color_index = 0
        self.last_gesture = "None"
        self.gesture_triggered = False

        self.hand_x = 0
        self.hand_y = 0
        self.prev_hand_x = 0
        self.prev_hand_y = 0
        self.current_gesture = "None"
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        self.hand_positions = []
        self.smoothing_window = 3
        self.smoothing_factor = 0.5
        
        self.fps_counter = FPSCounter()
        
        # Initialize modular components (code reuse)
        self.canvas_controller = None  # Will be set after canvas creation
        self.gesture_handler = GestureHandler()
        self.camera_thread_helper = None  # Will be set after detector init
        self.debug_mode = False
        
        # Gesture logging and debugging
        self.gesture_log = deque(maxlen=50)
        self.frame_count = 0
        self.timing_info = {
            "capture": 0,
            "process": 0,
            "drawing": 0,
            "display": 0
        }
        
        self.process_scale = PROCESSING_SCALE
        
        self.cap = None
        self.camera_running = False
        
        self.init_detector()
        self.create_widgets()
        self.start_camera()
        self.update_ui()
    
    def init_detector(self):
        was_running = self.camera_running
        if was_running:
            self.camera_running = False
            time.sleep(0.1)
        
        if self.detector:
            try:
                self.detector.cleanup()
            except:
                pass
        
        if self.detection_mode == "mediapipe":
            self.detector = MediaPipeDetector(show_debug=self.debug_mode)
        else:
            self.detector = CVDetector(show_debug=self.debug_mode)
        
        if was_running and self.cap is not None:
            self.camera_running = True
            threading.Thread(target=self.update_camera, daemon=True).start()
    
    def switch_detection_mode(self, mode):
        if mode == self.detection_mode:
            return
        self.detection_mode = mode
        self.init_detector()
    
    def toggle_debug(self):
        self.debug_mode = self.debug_var.get()
        # Update detector's debug overlay in real-time
        if self.detector:
            self.detector.show_debug_overlay = self.debug_mode

    def create_widgets(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        left_frame = tk.Frame(main_frame, width=400)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_frame.pack_propagate(False)
        
        tk.Label(left_frame, text="Camera Feed", font=("Arial", 12, "bold")).pack()
        
        mode_frame = tk.Frame(left_frame)
        mode_frame.pack(pady=5)
        tk.Label(mode_frame, text="Detection Mode:", font=("Arial", 9)).pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value=self.detection_mode)
        mode_combo = ttk.Combobox(mode_frame, textvariable=self.mode_var, 
                                   values=["mediapipe", "cv"], state="readonly", width=12)
        mode_combo.pack(side=tk.LEFT, padx=5)
        mode_combo.bind("<<ComboboxSelected>>", lambda e: self.switch_detection_mode(self.mode_var.get()))
        
        debug_frame = tk.Frame(left_frame)
        debug_frame.pack(pady=2)
        self.debug_var = tk.BooleanVar(value=False)
        tk.Checkbutton(debug_frame, text="Debug Mode", variable=self.debug_var, 
                      command=self.toggle_debug).pack()
        
        self.camera_label = tk.Label(left_frame, bg="black")
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        self.gesture_label = tk.Label(left_frame, text="Gesture: None", font=("Arial", 10))
        self.gesture_label.pack(pady=5)
        
        # Drawing state indicator
        self.state_label = tk.Label(left_frame, text="State: IDLE", 
                                    bg="gray", fg="white", font=('Arial', 10, 'bold'), width=20)
        self.state_label.pack(pady=2)
        
        color_frame = tk.Frame(left_frame)
        color_frame.pack(pady=5)
        tk.Label(color_frame, text="Current Color:", font=("Arial", 9)).pack(side=tk.LEFT)
        self.color_indicator = tk.Label(color_frame, text="  ", bg=self.brush_color, width=4, relief=tk.SOLID, borderwidth=2)
        self.color_indicator.pack(side=tk.LEFT, padx=5)
        
        # Drawing instructions panel
        instructions_frame = tk.LabelFrame(left_frame, text="Drawing Instructions", font=('Arial', 9, 'bold'))
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
        
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        toolbar = tk.Frame(right_frame, padx=5, pady=5)
        toolbar.pack(side=tk.LEFT, fill=tk.Y)
        
        colors = ["#000000", "#ffffff", "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FFA500", "#800080"]
        tk.Label(toolbar, text="Colors").pack(pady=(0,5))
        for c in colors:
            b = tk.Button(toolbar, bg=c, width=3, height=1, command=lambda col=c: self.set_color(col))
            b.pack(pady=2)
        
        tk.Button(toolbar, text="Choose Color", command=self.choose_color).pack(pady=8, fill=tk.X)
        
        tk.Label(toolbar, text="Size").pack(pady=(10,0))
        self.size_label = tk.Label(toolbar, text=f"{self.brush_size}")
        self.size_label.pack()
        self.size_scale = tk.Scale(toolbar, from_=1, to=50, orient=tk.HORIZONTAL, command=self.change_size)
        self.size_scale.set(self.brush_size)
        self.size_scale.pack()
        
        tk.Button(toolbar, text="Eraser", command=self.use_eraser).pack(pady=8, fill=tk.X)
        tk.Button(toolbar, text="Brush", command=self.use_brush).pack(pady=2, fill=tk.X)
        tk.Button(toolbar, text="Clear All", command=self.clear_canvas).pack(pady=12, fill=tk.X)
        tk.Button(toolbar, text="Save", command=self.save_canvas).pack(pady=2, fill=tk.X)
        
        self.canvas = tk.Canvas(right_frame, bg=self.bg_color, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Initialize CanvasController for code reuse
        self.canvas_controller = CanvasController(self.canvas)
        
        self.cursor_id = None

    def start_camera(self):
        def setup():
            self.cap = find_camera()
            if self.cap:
                setup_camera(self.cap, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)
                self.camera_running = True
                threading.Thread(target=self.update_camera, daemon=True).start()
            else:
                messagebox.showerror("Camera Error", "Could not open camera")
        
        threading.Thread(target=setup, daemon=True).start()

    def update_camera(self):
        while self.camera_running:
            if self.cap is None or not self.cap.isOpened():
                break
            
            try:
                t0 = time.perf_counter()
                
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                t1 = time.perf_counter()
                self.timing_info["capture"] = (t1 - t0) * 1000
                
                frame = cv2.flip(frame, 1)
                
                t2 = time.perf_counter()
                
                result = self.detector.process_frame(frame)
                
                t3 = time.perf_counter()
                self.timing_info["process"] = (t3 - t2) * 1000
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue
            
            if result['detected']:
                t_draw_start = time.perf_counter()
                
                raw_x = result['hand_x']
                raw_y = result['hand_y']
                
                if self.prev_hand_x == 0 and self.prev_hand_y == 0:
                    self.hand_x = raw_x
                    self.hand_y = raw_y
                else:
                    self.hand_x = self.prev_hand_x + self.smoothing_factor * (raw_x - self.prev_hand_x)
                    self.hand_y = self.prev_hand_y + self.smoothing_factor * (raw_y - self.prev_hand_y)
                
                self.prev_hand_x = self.hand_x
                self.prev_hand_y = self.hand_y
                
                t_draw_end = time.perf_counter()
                self.timing_info["drawing"] = (t_draw_end - t_draw_start) * 1000
                
                # Handle gestures from both MediaPipe and CV modes
                if 'gesture' in result:
                    gesture = result['gesture']
                    self.current_gesture = gesture
                    self.root.after(0, self.handle_gesture, gesture)
                else:
                    self.current_gesture = "None"
                
                # Update cursor and draw continuously
                self.root.after(0, self.update_canvas_cursor)
                if self.drawing_enabled:
                    self.root.after(0, self.draw_at_cursor)
            else:
                self.current_gesture = "None"
                self.drawing_enabled = False
                self.hand_positions.clear()
                self.timing_info["drawing"] = 0
            
            annotated = result['annotated_frame']
            
            # Update FPS counter (but don't draw it on frame)
            fps = self.fps_counter.update()
            
            display_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            with self.frame_lock:
                self.current_frame = display_frame

    def update_ui(self):
        if self.current_frame is not None:
            with self.frame_lock:
                frame = self.current_frame
            
            img = Image.fromarray(frame)
            
            # Downscale to fit 400px width while maintaining aspect ratio
            display_width = 380
            aspect_ratio = img.height / img.width
            display_height = int(display_width * aspect_ratio)
            img = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
            
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)
        
        self.gesture_label.configure(text=f"Gesture: {self.current_gesture}")
        
        self.root.after(16, self.update_ui)

    def handle_gesture(self, gesture):
        gesture_changed = (gesture != self.last_gesture)
        
        # Use GestureHandler for logging (code reuse)
        if self.gesture_handler:
            self.gesture_handler.process_gesture(gesture)
        
        # Log gesture event
        self.frame_count += 1
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] Frame {self.frame_count}: {gesture}"
        
        if gesture == "Draw":
            self.use_brush()
            self.drawing_enabled = True
            self.state_label.config(text="State: DRAWING", bg="green")
            log_entry += " -> DRAWING"
        elif gesture == "Erase":
            self.use_eraser()
            self.drawing_enabled = True
            self.state_label.config(text="State: ERASING", bg="orange")
            log_entry += " -> ERASING"
        elif gesture == "Cycle Color":
            if gesture_changed:
                self.cycle_color()
            self.drawing_enabled = False
            self.last_x, self.last_y = None, None
        elif gesture == "Clear":
            if gesture_changed:
                self.clear_canvas()
                log_entry += " -> CLEARED"
                self.state_label.config(text="State: CLEARED", bg="red")
            self.drawing_enabled = False
            self.last_x, self.last_y = None, None
        elif gesture == "Increase Size":
            if gesture_changed:
                new_size = min(50, self.brush_size + 1)
                self.size_scale.set(new_size)
            self.drawing_enabled = False
            self.last_x, self.last_y = None, None
        elif gesture == "Decrease Size":
            if gesture_changed:
                new_size = max(1, self.brush_size - 1)
                self.size_scale.set(new_size)
            self.drawing_enabled = False
            self.last_x, self.last_y = None, None
        else:
            self.drawing_enabled = False
            self.last_x, self.last_y = None, None
            if gesture == "None":
                self.state_label.config(text="State: IDLE", bg="gray")
        
        # Add to gesture log and print to console
        if gesture != "None" or gesture_changed:
            self.gesture_log.append(log_entry)
            print(log_entry)
        
        self.last_gesture = gesture

    def update_canvas_cursor(self):
        canvas_x = int(self.hand_x * self.canvas.winfo_width())
        canvas_y = int(self.hand_y * self.canvas.winfo_height())
        
        if self.cursor_id:
            self.canvas.delete(self.cursor_id)
        
        r = self.brush_size
        color = "red" if self.drawing_enabled else "blue"
        self.cursor_id = self.canvas.create_oval(
            canvas_x - r, canvas_y - r, canvas_x + r, canvas_y + r,
            outline=color, width=2
        )

    def draw_at_cursor(self):
        canvas_x = int(self.hand_x * self.canvas.winfo_width())
        canvas_y = int(self.hand_y * self.canvas.winfo_height())
        
        # Always show cursor position overlay (even when not drawing)
        self.canvas.delete("status_overlay")
        
        status_text = f"Pos: ({canvas_x}, {canvas_y}) | Mode: {self.detection_mode.upper()}"
        if self.drawing_enabled:
            tool = "ERASER" if self.eraser_on else "BRUSH"
            status_text += f" | {tool} ACTIVE ✓"
            indicator_color = "orange" if self.eraser_on else "lime"
        else:
            status_text += " | INACTIVE"
            indicator_color = "gray"
        
        # Draw semi-transparent background for status
        self.canvas.create_rectangle(
            5, 5, 450, 30, fill="black", stipple="gray50", tags="status_overlay"
        )
        self.canvas.create_text(
            10, 18, text=status_text, anchor="w", 
            fill="yellow", font=('Arial', 9, 'bold'), tags="status_overlay"
        )
        
        # Draw active cursor indicator with color coding
        self.canvas.create_oval(
            canvas_x - self.brush_size - 2, canvas_y - self.brush_size - 2,
            canvas_x + self.brush_size + 2, canvas_y + self.brush_size + 2,
            outline=indicator_color, width=3, tags="status_overlay"
        )
        
        if not self.drawing_enabled:
            return
        
        color = self.bg_color if self.eraser_on else self.brush_color
        
        # Reuse CanvasController methods for drawing (improved maintainability)
        if self.canvas_controller:
            if self.last_x is not None and self.last_y is not None:
                self.canvas_controller.draw_line(self.last_x, self.last_y, canvas_x, canvas_y, 
                                                 color, self.brush_size)
            else:
                self.canvas_controller.draw_point(canvas_x, canvas_y, color, self.brush_size)
        else:
            # Fallback to direct canvas operations
            if self.last_x is not None and self.last_y is not None:
                self.canvas.create_line(
                    self.last_x, self.last_y, canvas_x, canvas_y,
                    fill=color, width=self.brush_size,
                    capstyle=tk.ROUND, smooth=True
                )
            else:
                r = self.brush_size / 2
                self.canvas.create_oval(
                    canvas_x - r, canvas_y - r, canvas_x + r, canvas_y + r,
                    fill=color, outline=color
                )
        
        self.last_x, self.last_y = canvas_x, canvas_y

    def set_color(self, col):
        self.brush_color = col
        self.eraser_on = False
        self.color_indicator.configure(bg=col)

    def cycle_color(self):
        self.current_color_index = (self.current_color_index + 1) % len(self.color_palette)
        self.set_color(self.color_palette[self.current_color_index])

    def choose_color(self):
        col = colorchooser.askcolor(color=self.brush_color, title="Choose Color")
        if col and col[1]:
            self.set_color(col[1])

    def change_size(self, val):
        try:
            self.brush_size = int(val)
            self.size_label.configure(text=f"{self.brush_size}")
        except ValueError:
            pass

    def use_eraser(self):
        self.eraser_on = True

    def use_brush(self):
        self.eraser_on = False

    def clear_canvas(self):
        # Reuse CanvasController for maintainability
        if self.canvas_controller:
            self.canvas_controller.clear_canvas()
        else:
            self.canvas.delete("all")
            self.canvas.configure(bg=self.bg_color)
        self.last_x, self.last_y = None, None

    def save_canvas(self):
        file = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg")]
        )
        if not file:
            return
        try:
            self.root.update()
            ps_path = file + ".ps"
            self.canvas.postscript(file=ps_path, colormode='color')
            
            img = Image.open(ps_path)
            if img.mode == "RGBA":
                bg = Image.new("RGBA", img.size, self.bg_color)
                bg.paste(img, (0,0), img)
                img = bg.convert("RGB")
            else:
                img = img.convert("RGB")
            
            img.save(file)
            import os
            os.remove(ps_path)
            messagebox.showinfo("Saved", f"Image saved:\n{file}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save:\n{e}")

    def cleanup(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
        if self.detector:
            self.detector.cleanup()

def main():
    root = tk.Tk()
    app = GesturePaintApp(root, detection_mode="mediapipe")
    root.protocol("WM_DELETE_WINDOW", lambda: (app.cleanup(), root.destroy()))
    root.mainloop()

if __name__ == "__main__":
    main()
