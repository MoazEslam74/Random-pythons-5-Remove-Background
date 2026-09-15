import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO

class ObjectExtractorApp:
    lang = 'ar'
    LANG_DATA = {
        'ar': {"upload_button": "رفع صورة", "image_label": "لم يتم اختيار صورة بعد", "processing": "جاري المعالجة..."},
        'en': {"upload_button": "Upload Image", "image_label": "No image selected yet", "processing": "Processing..."}
    }
    
    def __init__(self, root):
        self.root = root
        self.root.title("أداة العزل الذكية")
        self.root.geometry("800x650")

        # Load the YOLO model
        self.model = YOLO('yolov8n.pt')
        
        # Variables for storing the image and object state
        self.current_image_path = None
        self.cv_image_rgb = None     # Store a clean copy of the original image
        self.detected_boxes = []     # List for storing coordinates of detected objects
        self.selected_box = None     # Store the currently clicked/selected box
        
        # Variables for coordinate mapping (mouse click to original image size)
        self.scale_x = 1.0
        self.scale_y = 1.0

        # --- Interface elements ---
        self.lang_toggle_btn = tk.Button(self.root, text="English", command=self.toggle_language, font=("Arial", 12))
        self.lang_toggle_btn.pack(pady=10)
        
        self.upload_btn = tk.Button(self.root, text=self.LANG_DATA[self.lang]["upload_button"], command=self.upload_image, font=("Arial", 14))
        self.upload_btn.pack(pady=10)

        self.image_label = tk.Label(self.root, text=self.LANG_DATA[self.lang]["image_label"], font=("Arial", 12))
        self.image_label.pack(expand=True)
        
        # Bind mouse click event to the image label
        self.image_label.bind("<Button-1>", self.on_image_click)

    def toggle_language(self):
        self.lang = 'en' if self.lang == 'ar' else 'ar'
        self.lang_toggle_btn.config(text="العربية" if self.lang == 'en' else "English")
        self.update_interface_language()

    def update_interface_language(self):
        self.upload_btn.config(text=self.LANG_DATA[self.lang]["upload_button"])
        if not self.cv_image_rgb is not None:
            self.image_label.config(text=self.LANG_DATA[self.lang]["image_label"])

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="اختر صورة" if self.lang == 'ar' else "Select Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.image_label.config(text=self.LANG_DATA[self.lang]["processing"], image="")
            self.root.update()
            
            self.process_and_display_image(file_path)

    def process_and_display_image(self, path):
        # 1. Read the image using OpenCV and keep a clean copy in memory
        cv_image = cv2.imread(path)
        self.cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB) 
        
        # 2. Pass the clean image to the model to detect objects
        results = self.model(self.cv_image_rgb)
        self.detected_boxes = []
        self.selected_box = None # Reset selection on new upload
        
        # 3. Create a copy for drawing bounding boxes so we don't ruin the clean original
        draw_img = self.cv_image_rgb.copy()
        
        for result in results:
            for box in result.boxes.xyxy: # Coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box[:4])
                self.detected_boxes.append((x1, y1, x2, y2))
                # Draw the initial boxes in green
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # 4. Display the image and calculate scale factors
        self.display_numpy_image(draw_img)

    def display_numpy_image(self, img_array):
        """Helper function to resize, calculate scale factors, and display an image array"""
        img_pil = Image.fromarray(img_array)
        orig_w, orig_h = img_pil.size
        
        # Resize preserving aspect ratio
        display_img = img_pil.copy()
        display_img.thumbnail((600, 450))
        disp_w, disp_h = display_img.size
        
        # Calculate scale factor (Original Dimension / Display Dimension)
        self.scale_x = orig_w / disp_w
        self.scale_y = orig_h / disp_h
        
        # Update Tkinter label
        self.tk_image = ImageTk.PhotoImage(display_img)
        self.image_label.config(image=self.tk_image, text="")

    def on_image_click(self, event):
        """Triggered when the user clicks on the image label"""
        # Ensure we have detected boxes and an image is loaded
        if not self.detected_boxes or self.cv_image_rgb is None:
            return
            
        # 1. Map the click coordinates from display size to original image size
        orig_x = int(event.x * self.scale_x)
        orig_y = int(event.y * self.scale_y)
        
        # 2. Check if the click falls within any detected bounding box
        clicked_box = None
        for box in self.detected_boxes:
            x1, y1, x2, y2 = box
            if x1 <= orig_x <= x2 and y1 <= orig_y <= y2:
                clicked_box = box
                break # Stop at the first box found
                
        # 3. If a box was clicked, highlight it
        if clicked_box:
            self.selected_box = clicked_box
            self.highlight_selected_box()

    def highlight_selected_box(self):
        """Redraws the image with the selected box in red and others in green"""
        draw_img = self.cv_image_rgb.copy()
        
        for box in self.detected_boxes:
            x1, y1, x2, y2 = box
            if box == self.selected_box:
                # Highlight the selected box in Red (RGB format) and make it thicker
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (255, 0, 0), 5)
            else:
                # Keep other boxes in Green
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
        self.display_numpy_image(draw_img)

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectExtractorApp(root)
    root.mainloop()