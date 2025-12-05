"""
Drawing state and cursor management
"""
import tkinter as tk


class DrawingManager:
    """Manages drawing state, cursor, and canvas operations"""
    
    def __init__(self, canvas):
        self.canvas = canvas
        self.last_x = None
        self.last_y = None
        self.cursor_id = None
        self.eraser_on = False
        self.brush_color = "#000000"
        self.bg_color = "white"
        self.brush_size = 5
        self.drawing_enabled = False
    
    def update_cursor_position(self, canvas_x, canvas_y, drawing_enabled=None):
        """Update cursor visualization on canvas"""
        if self.cursor_id:
            self.canvas.delete(self.cursor_id)
        
        if drawing_enabled is not None:
            self.drawing_enabled = drawing_enabled
        
        r = self.brush_size
        color = "red" if self.drawing_enabled else "blue"
        self.cursor_id = self.canvas.create_oval(
            canvas_x - r, canvas_y - r, canvas_x + r, canvas_y + r,
            outline=color, width=2
        )
    
    def draw_status_overlay(self, canvas_x, canvas_y, detection_mode, drawing_enabled=None):
        """Draw status overlay on canvas"""
        if drawing_enabled is not None:
            self.drawing_enabled = drawing_enabled
            
        self.canvas.delete("status_overlay")
        
        status_text = f"Pos: ({canvas_x}, {canvas_y}) | Mode: {detection_mode.upper()}"
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
    
    def draw_at_position(self, canvas_x, canvas_y):
        """Draw on canvas at given position"""
        if not self.drawing_enabled:
            return
        
        color = self.bg_color if self.eraser_on else self.brush_color
        
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
    
    def reset_drawing_position(self):
        """Reset last drawing position"""
        self.last_x = None
        self.last_y = None
    
    def set_drawing_enabled(self, enabled):
        """Enable or disable drawing"""
        self.drawing_enabled = enabled
        if not enabled:
            self.reset_drawing_position()
    
    def use_brush(self):
        """Switch to brush mode"""
        self.eraser_on = False
    
    def use_eraser(self):
        """Switch to eraser mode"""
        self.eraser_on = True
    
    def set_brush_color(self, color):
        """Set brush color"""
        self.brush_color = color
        self.eraser_on = False
    
    def set_brush_size(self, size):
        """Set brush size"""
        self.brush_size = size
    
    def clear_canvas(self):
        """Clear entire canvas"""
        self.canvas.delete("all")
        self.canvas.configure(bg=self.bg_color)
        self.reset_drawing_position()
