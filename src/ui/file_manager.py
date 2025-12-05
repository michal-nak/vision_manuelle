"""
File operations for saving canvas
"""
from tkinter import filedialog, messagebox
from PIL import Image
import os


class FileManager:
    """Handles file operations like saving canvas"""
    
    @staticmethod
    def save_canvas(root, canvas, bg_color):
        """Save canvas as image file"""
        file = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg")]
        )
        if not file:
            return
        
        try:
            root.update()
            ps_path = file + ".ps"
            canvas.postscript(file=ps_path, colormode='color')
            
            img = Image.open(ps_path)
            if img.mode == "RGBA":
                bg = Image.new("RGBA", img.size, bg_color)
                bg.paste(img, (0,0), img)
                img = bg.convert("RGB")
            else:
                img = img.convert("RGB")
            
            img.save(file)
            os.remove(ps_path)
            messagebox.showinfo("Saved", f"Image saved:\n{file}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save:\n{e}")
