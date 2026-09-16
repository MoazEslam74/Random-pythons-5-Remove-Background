import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
from rembg import remove # Added rembg

class ObjectExtractorApp:
    lang = 'ar'
    LANG_DATA = {
        'ar': {
            "upload_button": "رفع صورة", 
            "image_label": "لم يتم اختيار صورة بعد", 
            "processing": "جاري المعالجة...",
            "extract_button": "عزل الكائن المحدد",
            "extracting": "جاري العزل والحفظ...",
            "success": "تم حفظ الكائن المعزول بنجاح باسم: isolated_object.png"
        },
        'en': {
            "upload_button": "Upload Image", 
            "image_label": "No image selected yet", 
            "processing": "Processing...",
            "extract_button": "Extract Selected Object",
            "extracting": "Extracting and saving...",
            "success": "Isolated object saved successfully as: isolated_object.png"
        }
    }
    
    def __init__(self, root):
        self.root = root
        self.root.title("أداة العزل الذكية")
        self.root.geometry("800x750") # Increased height to accommodate the new button

        # Load the YOLO model
        self.model = YOLO('yolov8n.pt')
        
        # Variables for storing the image and object state
        self.current_image_path = None
        self.cv_image_rgb = None
        self.detected_boxes = []
        self.selected_box = None
        self.scale_x = 1.0
        self.scale_y = 1.0

        # --- Interface elements ---
        self.lang_toggle_btn = tk.Button(self.root, text="English", command=self.toggle_language, font=("Arial", 12))
        self.lang_toggle_btn.pack(pady=5)
        
        self.upload_btn = tk.Button(self.root, text=self.LANG_DATA[self.lang]["upload_button"], command=self.upload_image, font=("Arial", 14))
        self.upload_btn.pack(pady=10)

        # Extraction button (Hidden initially)
        self.extract_btn = tk.Button(self.root, text=self.LANG_DATA[self.lang]["extract_button"], command=self.extract_and_save, font=("Arial", 14), bg="lightblue")
        
        self.image_label = tk.Label(self.root, text=self.LANG_DATA[self.lang]["image_label"], font=("Arial", 12))
        self.image_label.pack(expand=True, pady=10)
        
        self.image_label.bind("<Button-1>", self.on_image_click)

    def toggle_language(self):
        self.lang = 'en' if self.lang == 'ar' else 'ar'
        self.lang_toggle_btn.config(text="العربية" if self.lang == 'en' else "English")
        self.update_interface_language()

    def update_interface_language(self):
        self.upload_btn.config(text=self.LANG_DATA[self.lang]["upload_button"])
        self.extract_btn.config(text=self.LANG_DATA[self.lang]["extract_button"])
        if self.cv_image_rgb is None:
            self.image_label.config(text=self.LANG_DATA[self.lang]["image_label"])

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="اختر صورة" if self.lang == 'ar' else "Select Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.image_label.config(text=self.LANG_DATA[self.lang]["processing"], image="")
            self.extract_btn.pack_forget() # Hide extract button on new upload
            self.root.update()
            
            self.process_and_display_image(file_path)

    def process_and_display_image(self, path):
        cv_image = cv2.imread(path)
        self.cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB) 
        
        results = self.model(self.cv_image_rgb,conf=0.15)  # Adjust confidence threshold as needed
        self.detected_boxes = []
        self.selected_box = None 
        
        draw_img = self.cv_image_rgb.copy()
        
        for result in results:
            for box in result.boxes.xyxy: 
                x1, y1, x2, y2 = map(int, box[:4])
                self.detected_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        self.display_numpy_image(draw_img)

    def display_numpy_image(self, img_array):
        img_pil = Image.fromarray(img_array)
        orig_w, orig_h = img_pil.size
        
        display_img = img_pil.copy()
        display_img.thumbnail((600, 450))
        disp_w, disp_h = display_img.size
        
        self.scale_x = orig_w / disp_w
        self.scale_y = orig_h / disp_h
        
        self.tk_image = ImageTk.PhotoImage(display_img)
        self.image_label.config(image=self.tk_image, text="")

    def on_image_click(self, event):
        if not self.detected_boxes or self.cv_image_rgb is None:
            return
            
        orig_x = int(event.x * self.scale_x)
        orig_y = int(event.y * self.scale_y)
        
        clicked_box = None
        for box in self.detected_boxes:
            x1, y1, x2, y2 = box
            if x1 <= orig_x <= x2 and y1 <= orig_y <= y2:
                clicked_box = box
                break 
                
        if clicked_box:
            self.selected_box = clicked_box
            self.highlight_selected_box()
            # Show the extraction button once a box is selected
            self.extract_btn.pack(after=self.upload_btn, pady=10)

    def highlight_selected_box(self):
        draw_img = self.cv_image_rgb.copy()
        
        for box in self.detected_boxes:
            x1, y1, x2, y2 = box
            if box == self.selected_box:
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (255, 0, 0), 5)
            else:
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
        self.display_numpy_image(draw_img)

    def extract_and_save(self):
        if self.selected_box is None or self.cv_image_rgb is None:
            return
            
        self.extract_btn.config(text=self.LANG_DATA[self.lang]["extracting"], state=tk.DISABLED)
        self.root.update()

        try:
            img = self.cv_image_rgb.copy()
            x1, y1, x2, y2 = self.selected_box

            # --- New adjustment: calculate a safe padding margin ---
            # Expand the box by 15% of the object's width and height
            padding_percent = 0.15
            w = x2 - x1
            h = y2 - y1
            
            pad_x = int(w * padding_percent)
            pad_y = int(h * padding_percent)
            
            # Calculate the new coordinates while ensuring they do not exceed the image bounds
            img_h, img_w = img.shape[:2]
            px1 = max(0, x1 - pad_x)
            py1 = max(0, y1 - pad_y)
            px2 = min(img_w, x2 + pad_x)
            py2 = min(img_h, y2 + pad_y)

            # Blur the image
            blurred_img = cv2.GaussianBlur(img, (99, 99), 30)

            # Extract the sharp region using the expanded coordinates
            sharp_roi = img[py1:py2, px1:px2]

            # Merge the sharp region back
            manipulated_img = blurred_img.copy()
            manipulated_img[py1:py2, px1:px2] = sharp_roi

            manipulated_pil = Image.fromarray(manipulated_img)
            final_output = remove(manipulated_pil)
            
            final_output.save("isolated_object.png")
            messagebox.showinfo("Success", self.LANG_DATA[self.lang]["success"])
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            
        finally:
            self.extract_btn.config(text=self.LANG_DATA[self.lang]["extract_button"], state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectExtractorApp(root)
    root.mainloop()