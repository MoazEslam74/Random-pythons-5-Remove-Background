import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO # YOLO added

class ObjectExtractorApp:
    lang='ar'
    LANG_DATA={
        'ar':{"upload_button": "رفع صورة", "image_label": "لم يتم اختيار صورة بعد", "processing": "جاري المعالجة..."},
        'en':{"upload_button": "Upload Image", "image_label": "No image selected yet", "processing": "Processing..."}
    }
    
    def __init__(self, root):
        self.root = root
        self.root.title("أداة العزل الذكية")
        self.root.geometry("800x600")

        # Load the YOLO model (yolov8n.pt weights will be downloaded automatically the first time)
        self.model = YOLO('yolov8n.pt')
        
        # Variables for storing the image and object state
        self.current_image_path = None
        self.original_image = None
        self.detected_boxes = [] # List for storing the coordinates of detected objects

        # --- Interface elements ---
        self.lang_toggle_btn = tk.Button(self.root, text="English", command=self.toggle_language, font=("Arial", 12))
        self.lang_toggle_btn.pack(pady=10)
        
        self.upload_btn = tk.Button(self.root, text=self.LANG_DATA[self.lang]["upload_button"], command=self.upload_image, font=("Arial", 14))
        self.upload_btn.pack(pady=20)

        self.image_label = tk.Label(self.root, text=self.LANG_DATA[self.lang]["image_label"], font=("Arial", 12))
        self.image_label.pack(expand=True)

    def toggle_language(self):
        self.lang = 'en' if self.lang == 'ar' else 'ar'
        self.lang_toggle_btn.config(text="العربية" if self.lang == 'en' else "English")
        self.update_interface_language()

    def update_interface_language(self):
        self.upload_btn.config(text=self.LANG_DATA[self.lang]["upload_button"])
        if not self.current_image_path:
            self.image_label.config(text=self.LANG_DATA[self.lang]["image_label"])

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="اختر صورة" if self.lang == 'ar' else "Select Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.image_label.config(text=self.LANG_DATA[self.lang]["processing"], image="")
            self.root.update() # Update the interface to show the processing text
            
            # Pass the image to the YOLO model
            self.process_and_display_image(file_path)

    def process_and_display_image(self, path):
        # 1. Read the image using OpenCV
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert colors for PIL and YOLO
        
        # 2. Pass the image to the model to detect objects
        results = self.model(img_rgb)
        self.detected_boxes = [] # Clear the old list
        
        # 3. Draw boxes around detected objects
        for result in results:
            for box in result.boxes.xyxy: # Coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box[:4])
                self.detected_boxes.append((x1, y1, x2, y2))
                
                # Draw the box in green with a thickness of 3 pixels
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # 4. Convert the resulting image for display in Tkinter
        self.original_image = Image.fromarray(img_rgb)
        display_img = self.original_image.copy()
        
        # Preserve the aspect ratio when resizing
        display_img.thumbnail((600, 450))
        self.tk_image = ImageTk.PhotoImage(display_img)
        
        self.image_label.config(image=self.tk_image, text="")

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectExtractorApp(root)
    root.mainloop()