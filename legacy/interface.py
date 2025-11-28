import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox
from PIL import Image, ImageGrab, ImageOps, ImageTk
import os
import sys
import time

class PaintApp:
    def __init__(self, root):
        self.root = root
        root.title("Mini Paint")
        root.geometry("900x600")
        root.minsize(600, 400)

        # état du pinceau
        self.brush_color = "#000000"
        self.bg_color = "white"
        self.brush_size = 5
        self.eraser_on = False
        self.last_x = None
        self.last_y = None

        # interface
        self.create_widgets()
        self.bind_events()

    def create_widgets(self):
        # cadre outils
        toolbar = tk.Frame(self.root, padx=5, pady=5)
        toolbar.pack(side=tk.LEFT, fill=tk.Y)

        # couleurs prédéfinies
        colors = ["#000000", "#ffffff", "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FFA500", "#800080"]
        tk.Label(toolbar, text="Couleurs").pack(pady=(0,5))
        for c in colors:
            b = tk.Button(toolbar, bg=c, width=3, height=1, command=lambda col=c: self.set_color(col))
            b.pack(pady=2)

        # sélection couleur
        tk.Button(toolbar, text="Choisir couleur", command=self.choose_color).pack(pady=8, fill=tk.X)

        # taille du pinceau
        tk.Label(toolbar, text="Taille").pack(pady=(10,0))
        self.size_scale = tk.Scale(toolbar, from_=1, to=50, orient=tk.HORIZONTAL, command=self.change_size)
        self.size_scale.set(self.brush_size)
        self.size_scale.pack()

        # outils
        tk.Button(toolbar, text="Gomme", command=self.use_eraser).pack(pady=8, fill=tk.X)
        tk.Button(toolbar, text="Pinceau", command=self.use_brush).pack(pady=2, fill=tk.X)
        tk.Button(toolbar, text="Effacer tout", command=self.clear_canvas).pack(pady=12, fill=tk.X)
        tk.Button(toolbar, text="Ouvrir image...", command=self.open_image).pack(pady=2, fill=tk.X)
        tk.Button(toolbar, text="Enregistrer", command=self.save_canvas).pack(pady=2, fill=tk.X)

        # zone de dessin
        self.canvas = tk.Canvas(self.root, bg=self.bg_color, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def bind_events(self):
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_paint)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        # raccourcis
        self.root.bind("<Control-s>", lambda e: self.save_canvas())
        self.root.bind("<Control-o>", lambda e: self.open_image())
        self.root.bind("e", lambda e: self.use_eraser())
        self.root.bind("b", lambda e: self.use_brush())
        self.root.bind("c", lambda e: self.clear_canvas())

        self.root.bind_all("r", lambda e: self.set_color("#FF0000"))

    def set_color(self, col):
        self.brush_color = col
        self.eraser_on = False

    def choose_color(self):
        col = colorchooser.askcolor(color=self.brush_color, title="Choisir une couleur")
        if col and col[1]:
            self.set_color(col[1])

    def change_size(self, val):
        try:
            self.brush_size = int(val)
        except ValueError:
            pass

    def use_eraser(self):
        self.eraser_on = True

    def use_brush(self):
        self.eraser_on = False

    def on_button_press(self, event):
        self.last_x, self.last_y = event.x, event.y
        # petit point quand simple clic
        color = self.bg_color if self.eraser_on else self.brush_color
        r = self.brush_size / 2
        self.canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r,
                                fill=color, outline=color, width=0)

    def on_paint(self, event):
        x, y = event.x, event.y
        color = self.bg_color if self.eraser_on else self.brush_color
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                    fill=color, width=self.brush_size,
                                    capstyle=tk.ROUND, smooth=True, splinesteps=36)
        self.last_x, self.last_y = x, y

    def on_button_release(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        # assure couleur de fond uniforme
        self.canvas.configure(bg=self.bg_color)

    def open_image(self):
        path = filedialog.askopenfilename(title="Ouvrir une image",
                                          filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif"), ("Tous", "*.*")])
        if not path:
            return
        try:
            img = Image.open(path)
            # redimensionner pour tenir dans le canvas actuel (conserver ratio)
            cw = max(100, self.canvas.winfo_width())
            ch = max(100, self.canvas.winfo_height())
            img.thumbnail((cw-10, ch-10), Image.LANCZOS)
            self.tkimg = ImageTk.PhotoImage(img)
            self.canvas.create_image(5, 5, anchor=tk.NW, image=self.tkimg)
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'ouvrir l'image:\n{e}")

    def save_canvas(self):
        # sauvegarde via postscript puis conversion avec Pillow pour avoir PNG/JPG
        file = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg")])
        if not file:
            return
        try:
            # mettre à jour l'affichage
            self.root.update()
            # crée un fichier postscript temporaire
            ps_path = file + ".ps"
            self.canvas.postscript(file=ps_path, colormode='color')

            # ouvrir et convertir
            img = Image.open(ps_path)
            # postscript n'inclut pas le fond, on l'applique si nécessaire
            if img.mode == "RGBA":
                bg = Image.new("RGBA", img.size, self.bg_color)
                bg.paste(img, (0,0), img)
                img = bg.convert("RGB")
            else:
                img = img.convert("RGB")
            # enregistrer final
            img.save(file)
            os.remove(ps_path)
            messagebox.showinfo("Enregistré", f"Image enregistrée:\n{file}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Echec enregistrement:\n{e}")

def main():
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()