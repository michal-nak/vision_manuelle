"""
Canvas drawing logic separated from UI
"""
import tkinter as tk


class CanvasController:
    """Handles all canvas drawing operations"""
    
    def __init__(self, canvas):
        self.canvas = canvas
        self.brush_size = 5
        self.brush_color = "black"
        self.bg_color = "white"
        self.eraser_on = False
        self.drawing_enabled = False
        
        self.last_x = None
        self.last_y = None
        self.cursor_id = None
    
    def use_brush(self):
        """Switch to brush mode"""
        self.eraser_on = False
    
    def use_eraser(self):
        """Switch to eraser mode"""
        self.eraser_on = True
    
    def clear_canvas(self):
        """Clear entire canvas"""
        self.canvas.delete("all")
        self.last_x = None
        self.last_y = None
    
    def set_brush_size(self, size):
        """Set brush/eraser size"""
        self.brush_size = size
    
    def set_brush_color(self, color):
        """Set brush color"""
        self.brush_color = color
        self.eraser_on = False
    
    def draw_line(self, x1, y1, x2, y2, color=None, width=None):
        """Draw a line on canvas"""
        if color is None:
            color = self.bg_color if self.eraser_on else self.brush_color
        if width is None:
            width = self.brush_size
        self.canvas.create_line(
            x1, y1, x2, y2,
            fill=color,
            width=width,
            capstyle=tk.ROUND,
            smooth=True
        )
    
    def draw_point(self, x, y, color=None, size=None):
        """Draw a single point"""
        if color is None:
            color = self.bg_color if self.eraser_on else self.brush_color
        if size is None:
            size = self.brush_size
        r = size / 2
        self.canvas.create_oval(
            x - r, y - r, x + r, y + r,
            fill=color, outline=color
        )
    
    def update_cursor(self, x, y):
        """Update cursor position indicator"""
        if self.cursor_id:
            self.canvas.delete(self.cursor_id)
        
        r = self.brush_size
        color = "red" if self.drawing_enabled else "blue"
        self.cursor_id = self.canvas.create_oval(
            x - r, y - r, x + r, y + r,
            outline=color, width=2
        )
    
    def draw_status_overlay(self, x, y, mode, is_drawing):
        """Draw status overlay on canvas"""
        self.canvas.delete("status_overlay")
        
        status_text = f"Pos: ({x}, {y}) | Mode: {mode.upper()}"
        if is_drawing:
            tool = "ERASER" if self.eraser_on else "BRUSH"
            status_text += f" | {tool} ACTIVE ✓"
            indicator_color = "orange" if self.eraser_on else "lime"
        else:
            status_text += " | INACTIVE"
            indicator_color = "gray"
        
        # Semi-transparent background
        self.canvas.create_rectangle(
            5, 5, 450, 30, fill="black",
            stipple="gray50", tags="status_overlay"
        )
        self.canvas.create_text(
            10, 18, text=status_text, anchor="w",
            fill="yellow", font=('Arial', 9, 'bold'),
            tags="status_overlay"
        )
        
        # Cursor indicator circle
        self.canvas.create_oval(
            x - self.brush_size - 2, y - self.brush_size - 2,
            x + self.brush_size + 2, y + self.brush_size + 2,
            outline=indicator_color, width=3, tags="status_overlay"
        )
    
    def save_to_file(self, filename):
        """Save canvas to file (PostScript)"""
        self.canvas.postscript(file=filename)
