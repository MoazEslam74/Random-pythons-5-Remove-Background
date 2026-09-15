import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ObjectExtractorApp:
    lang='ar'
    LANG_DATA={
        'ar':{"upload_button": "رفع صورة", "image_label": "لم يتم اختيار صورة بعد"},
        'en':{"upload_button": "Upload Image", "image_label": "No image selected yet"}
    }
    def toggle_language(self):
        self.lang = 'en' if self.lang == 'ar' else 'ar'
        self.update_interface_language()

    def update_interface_language(self):
        self.upload_btn.config(text=self.LANG_DATA[self.lang]["upload_button"])
        self.image_label.config(text=self.LANG_DATA[self.lang]["image_label"])    
    def __init__(self, root):
        self.root = root
        self.root.title("أداة العزل الذكية")
        self.root.geometry("800x600")

        # Variable for storing the current image path
        self.current_image_path = None
        self.original_image = None

        # --- Interface elements ---
        # languuage toggle button
        self.lang_toggle_btn = tk.Button(self.root, text="English", command=self.toggle_language, font=("Arial", 12))
        self.lang_toggle_btn.pack(pady=10)
        # 1. Upload image button
        self.upload_btn = tk.Button(self.root, text=self.LANG_DATA[self.lang]["upload_button"], command=self.upload_image, font=("Arial", 14))
        self.upload_btn.pack(pady=20)

        # 2. Image display area
        self.image_label = tk.Label(self.root, text=self.LANG_DATA[self.lang]["image_label"], font=("Arial", 12))
        self.image_label.pack(expand=True)

    def upload_image(self):
        # Open the file selection dialog
        file_path = filedialog.askopenfilename(
            title="اختر صورة",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            
            # The segmentation function will be called here later
            # print("Passing the image to the segmentation model...")

    def display_image(self, path):
        # Open the image using Pillow
        self.original_image = Image.open(path)
        
        # Resize the image to fit the interface while preserving its dimensions
        display_img = self.original_image.copy()
        display_img.thumbnail((600, 450))
        
        # Convert it to a format supported by Tkinter
        self.tk_image = ImageTk.PhotoImage(display_img)
        
        # Update the interface to display the image
        self.image_label.config(image=self.tk_image, text="")

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectExtractorApp(root)
    root.mainloop()