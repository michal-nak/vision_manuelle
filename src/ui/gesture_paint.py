import tkinter as tk
from tkinter import colorchooser
import threading
import time
from ..detectors import MediaPipeDetector, CVDetector
from ..core.config import (
    UI_WINDOW_WIDTH, UI_WINDOW_HEIGHT,
    COLOR_PALETTE
)

# Import modular components
from .ui_components import UIComponents
from .gesture_state import GestureStateManager
from .camera_manager import CameraManager
from .drawing_manager import DrawingManager
from .file_manager import FileManager

class GesturePaintApp:
    def __init__(self, root, detection_mode="mediapipe", show_debug=False):
        self.root = root
        root.title("Gesture Paint - Hand Control")
        root.geometry(f"{UI_WINDOW_WIDTH}x{UI_WINDOW_HEIGHT}")
        root.minsize(800, 400)
        
        # Detection settings
        self.detection_mode = detection_mode
        self.detector = None
        self.debug_mode = show_debug
        
        # Color and brush settings
        self.bg_color = "white"
        self.color_palette = COLOR_PALETTE
        self.current_color_index = 0
        
        # Hand tracking
        self.hand_x = 0
        self.hand_y = 0
        self.prev_hand_x = 0
        self.prev_hand_y = 0
        self.smoothing_factor = 0.5
        # Coordinate mapping margin (allows reaching edges when hand is slightly off-screen)
        self.coord_margin = 0.15  # 15% margin on each side
        
        # Initialize managers
        self.gesture_state = GestureStateManager()
        self.camera_manager = CameraManager()
        self.drawing_manager = None  # Will be set after canvas creation
        
        # UI components
        self.camera_label = None
        self.gesture_label = None
        self.state_label = None
        self.color_indicator = None
        self.size_label = None
        self.size_scale = None
        self.canvas = None
        
        # Tkinter variables
        self.mode_var = None
        self.debug_var = None
        
        self.init_detector()
        self.create_widgets()
        self.start_camera()
        self.update_ui()
    
    def init_detector(self):
        was_running = self.camera_manager.is_running()
        
        # Clean up old detector first
        if self.detector:
            try:
                self.detector.cleanup()
            except:
                pass
            self.detector = None
        
        if was_running:
            # Fully release camera when switching modes to reset MediaPipe timestamps
            self.camera_manager.stop_camera(release_camera=True)
            time.sleep(0.3)  # Increased delay for proper camera release
        
        # Create new detector
        if self.detection_mode == "mediapipe":
            self.detector = MediaPipeDetector(show_debug=self.debug_mode)
        else:
            self.detector = CVDetector(show_debug=self.debug_mode)
        
        if was_running:
            # Properly restart camera after detector change
            self.start_camera()
    
    def switch_detection_mode(self, mode):
        if mode == self.detection_mode:
            return
        self.detection_mode = mode
        self.init_detector()
    
    def toggle_debug(self):
        self.debug_mode = self.debug_var.get()
        if self.detector:
            self.detector.show_debug_overlay = self.debug_mode
    
    def map_to_canvas(self, normalized_x, normalized_y):
        """Map normalized coordinates (0-1) to canvas with margin for easier edge access"""
        # Compress the camera's 0-1 range to use only the middle portion of the canvas
        # Then scale it back up to fill the canvas, effectively creating a margin
        # For example, with 0.15 margin:
        # Camera 0.0 -> Canvas edge (reaching edge requires hand at camera 0.15)
        # Camera 0.15 -> Canvas 0
        # Camera 0.85 -> Canvas max
        # Camera 1.0 -> Canvas edge (reaching edge requires hand at camera 0.85)
        
        # Scale and shift: map [margin, 1-margin] input to [0, 1] output
        scaled_x = (normalized_x - self.coord_margin) / (1 - 2 * self.coord_margin)
        scaled_y = (normalized_y - self.coord_margin) / (1 - 2 * self.coord_margin)
        
        # Clamp to canvas bounds
        canvas_x = max(0, min(int(scaled_x * self.canvas.winfo_width()), self.canvas.winfo_width() - 1))
        canvas_y = max(0, min(int(scaled_y * self.canvas.winfo_height()), self.canvas.winfo_height() - 1))
        
        return canvas_x, canvas_y

    def create_widgets(self):
        # Create main layout
        main_frame, left_frame, right_frame = UIComponents.create_main_layout(self.root)
        
        # Create Tkinter variables
        self.mode_var = tk.StringVar(value=self.detection_mode)
        self.debug_var = tk.BooleanVar(value=False)
        
        # Create camera panel
        self.camera_label = UIComponents.create_camera_panel(
            left_frame, self.mode_var, self.debug_var, self.detection_mode,
            lambda e: self.switch_detection_mode(self.mode_var.get()),
            self.toggle_debug
        )
        
        # Create gesture info panel
        self.gesture_label, self.state_label = UIComponents.create_gesture_info_panel(left_frame)
        
        # Create color indicator
        self.color_indicator = UIComponents.create_color_indicator(left_frame, COLOR_PALETTE[0])
        
        # Create instructions panel
        UIComponents.create_instructions_panel(left_frame)
        
        # Create toolbar
        self.size_label, self.size_scale = UIComponents.create_toolbar(
            right_frame,
            self.set_color,
            self.choose_color,
            self.change_size,
            self.clear_canvas,
            self.save_canvas,
            5  # initial brush size
        )
        
        # Create canvas
        self.canvas = UIComponents.create_canvas(right_frame, self.bg_color)
        
        # Initialize drawing manager
        self.drawing_manager = DrawingManager(self.canvas)
        self.drawing_manager.set_brush_color(COLOR_PALETTE[0])

    def start_camera(self):
        from tkinter import messagebox
        success = self.camera_manager.start_camera(
            lambda msg: messagebox.showerror("Camera Error", msg)
        )
        if success:
            threading.Thread(target=self.update_camera, daemon=True).start()

    def update_camera(self):
        while self.camera_manager.is_running():
            frame = self.camera_manager.capture_frame()
            if frame is None:
                continue
            
            try:
                t0 = time.perf_counter()
                result = self.detector.process_frame(frame)
                t1 = time.perf_counter()
                self.camera_manager.update_timing("process", (t1 - t0) * 1000)
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue
            
            if result['detected']:
                t_draw_start = time.perf_counter()
                
                # Smooth hand position
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
                self.camera_manager.update_timing("drawing", (t_draw_end - t_draw_start) * 1000)
                
                # Handle gestures
                if 'gesture' in result:
                    gesture = result['gesture']
                    self.gesture_state.current_gesture = gesture
                    self.root.after(0, self.handle_gesture, gesture)
                else:
                    self.gesture_state.current_gesture = "None"
                
                # Update cursor and draw
                self.root.after(0, self.update_canvas_cursor)
                if self.gesture_state.is_drawing_enabled():
                    self.root.after(0, self.draw_at_cursor)
            else:
                self.gesture_state.current_gesture = "None"
                self.gesture_state.enable_drawing(False)
                self.camera_manager.update_timing("drawing", 0)
            
            # Update frame for display
            self.camera_manager.update_frame(result['annotated_frame'])

    def update_ui(self):
        frame = self.camera_manager.get_current_frame()
        if frame is not None:
            imgtk = CameraManager.prepare_frame_for_display(frame)
            if imgtk:
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)
        
        self.gesture_label.configure(text=f"Gesture: {self.gesture_state.current_gesture}")
        
        self.root.after(16, self.update_ui)

    def handle_gesture(self, gesture):
        gesture_changed = (gesture != self.gesture_state.last_gesture)
        
        # Log gesture event
        self.gesture_state.frame_count += 1
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] Frame {self.gesture_state.frame_count}: {gesture}"
        
        if gesture == "Draw":
            self.drawing_manager.use_brush()
            self.gesture_state.enable_drawing(True)
            self.state_label.config(text="State: DRAWING", bg="green")
            log_entry += " -> DRAWING"
        elif gesture == "Erase":
            self.drawing_manager.use_eraser()
            self.gesture_state.enable_drawing(True)
            self.state_label.config(text="State: ERASING", bg="orange")
            log_entry += " -> ERASING"
        elif gesture == "Cycle Color":
            if gesture_changed:
                self.cycle_color()
            self.gesture_state.enable_drawing(False)
            self.drawing_manager.reset_drawing_position()
        elif gesture == "Clear":
            if gesture_changed:
                self.clear_canvas()
                log_entry += " -> CLEARED"
                self.state_label.config(text="State: CLEARED", bg="red")
            self.gesture_state.enable_drawing(False)
            self.drawing_manager.reset_drawing_position()
        elif gesture == "Increase Size":
            if gesture_changed:
                new_size = min(50, self.drawing_manager.brush_size + 1)
                self.size_scale.set(new_size)
            self.gesture_state.enable_drawing(False)
            self.drawing_manager.reset_drawing_position()
        elif gesture == "Decrease Size":
            if gesture_changed:
                new_size = max(1, self.drawing_manager.brush_size - 1)
                self.size_scale.set(new_size)
            self.gesture_state.enable_drawing(False)
            self.drawing_manager.reset_drawing_position()
        else:
            self.gesture_state.enable_drawing(False)
            self.drawing_manager.reset_drawing_position()
            if gesture == "None":
                self.state_label.config(text="State: IDLE", bg="gray")
        
        # Add to gesture log and print to console (only in debug mode)
        if gesture != "None" or gesture_changed:
            self.gesture_state.log_gesture(log_entry, debug_mode=self.debug_mode)
            if self.debug_mode:
                print(log_entry)
        
        self.gesture_state.set_last_gesture(gesture)

    def update_canvas_cursor(self):
        canvas_x, canvas_y = self.map_to_canvas(self.hand_x, self.hand_y)
        
        self.drawing_manager.update_cursor_position(
            canvas_x, canvas_y, self.gesture_state.is_drawing_enabled()
        )

    def draw_at_cursor(self):
        canvas_x, canvas_y = self.map_to_canvas(self.hand_x, self.hand_y)
        
        # Draw status overlay
        self.drawing_manager.draw_status_overlay(
            canvas_x, canvas_y, self.detection_mode,
            self.gesture_state.is_drawing_enabled()
        )
        
        if not self.gesture_state.is_drawing_enabled():
            return
        
        # Use drawing manager
        self.drawing_manager.draw_at_position(canvas_x, canvas_y)

    def set_color(self, col):
        self.drawing_manager.set_brush_color(col)
        self.drawing_manager.use_brush()
        self.color_indicator.configure(bg=col)

    def cycle_color(self):
        self.current_color_index = (self.current_color_index + 1) % len(self.color_palette)
        self.set_color(self.color_palette[self.current_color_index])

    def choose_color(self):
        col = colorchooser.askcolor(color=self.drawing_manager.brush_color, title="Choose Color")
        if col and col[1]:
            self.set_color(col[1])

    def change_size(self, val):
        try:
            new_size = int(val)
            self.drawing_manager.set_brush_size(new_size)
            self.size_label.configure(text=f"{new_size}")
        except ValueError:
            pass

    def use_eraser(self):
        self.drawing_manager.use_eraser()

    def use_brush(self):
        self.drawing_manager.use_brush()

    def clear_canvas(self):
        self.drawing_manager.clear_canvas()

    def save_canvas(self):
        FileManager.save_canvas(self.root, self.canvas, self.bg_color)

    def cleanup(self):
        self.camera_manager.cleanup()
        if self.detector:
            self.detector.cleanup()

def main():
    root = tk.Tk()
    app = GesturePaintApp(root, detection_mode="mediapipe")
    root.protocol("WM_DELETE_WINDOW", lambda: (app.cleanup(), root.destroy()))
    root.mainloop()

if __name__ == "__main__":
    main()
