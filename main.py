import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
from rembg import remove

class ObjectExtractorApp:
    lang = 'ar'
    LANG_DATA = {
        'ar': {
            "upload_button": "رفع صورة", 
            "image_label": "لم يتم اختيار صورة بعد", 
            "processing": "جاري المعالجة...",
            "extract_button": "عزل الكائن المحدد",
            "extracting": "جاري العزل...",
            "success": "تم العزل بنجاح وحفظ الصورة!",
            "manual_mode_btn": "وضع التحديد: يدوي",
            "auto_mode_btn": "وضع التحديد: تلقائي",
            "input_title": "الصورة الأصلية (للتحديد)",
            "output_title": "النتيجة النهائية",
            "output_placeholder": "ستظهر النتيجة هنا"
        },
        'en': {
            "upload_button": "Upload Image", 
            "image_label": "No image selected yet", 
            "processing": "Processing...",
            "extract_button": "Extract Selected",
            "extracting": "Extracting...",
            "success": "Extracted and saved successfully!",
            "manual_mode_btn": "Mode: Manual",
            "auto_mode_btn": "Mode: Auto (YOLO)",
            "input_title": "Input Image",
            "output_title": "Final Result",
            "output_placeholder": "Result will appear here"
        }
    }
    
    def __init__(self, root):
        self.root = root
        self.root.title("أداة العزل الذكية | Smart Extractor")
        self.root.geometry("1100x700") 
        self.root.configure(bg="#f0f2f5") # Modern light background

        # Load the YOLO model
        self.model = YOLO('yolov8n.pt')
        
        # Variables for storing the image and object state
        self.current_image_path = None
        self.cv_image_rgb = None
        self.detected_boxes = []
        self.selected_box = None
        self.scale_x = 1.0
        self.scale_y = 1.0

        # Variables for manual drawing
        self.is_manual_mode = False
        self.start_x = None
        self.start_y = None
        self.current_rect = None

        self.setup_ui()

    def setup_ui(self):
        """Builds the modern GUI layout"""
        # Styling variables
        btn_font = ("Helvetica", 11, "bold")
        title_font = ("Helvetica", 13, "bold")
        
        # --- 1. Top Control Panel ---
        self.control_frame = tk.Frame(self.root, bg="#ffffff", bd=1, relief="ridge", pady=10, padx=10)
        self.control_frame.pack(fill=tk.X, pady=(10, 5), padx=20)
        
        # Left side controls
        self.lang_toggle_btn = tk.Button(self.control_frame, text="English", command=self.toggle_language, font=btn_font, bg="#e4e6eb", relief="flat", padx=10)
        self.lang_toggle_btn.pack(side=tk.LEFT, padx=5)
        
        self.mode_toggle_btn = tk.Button(self.control_frame, text=self.LANG_DATA[self.lang]["manual_mode_btn"], command=self.toggle_mode, font=btn_font, bg="#e4e6eb", relief="flat", padx=10)
        self.mode_toggle_btn.pack(side=tk.LEFT, padx=5)

        # Right side controls
        self.extract_btn = tk.Button(self.control_frame, text=self.LANG_DATA[self.lang]["extract_button"], command=self.extract_and_save, font=btn_font, bg="#4CAF50", fg="white", relief="flat", padx=15, state=tk.DISABLED)
        self.extract_btn.pack(side=tk.RIGHT, padx=5)

        self.upload_btn = tk.Button(self.control_frame, text=self.LANG_DATA[self.lang]["upload_button"], command=self.upload_image, font=btn_font, bg="#007bff", fg="white", relief="flat", padx=15)
        self.upload_btn.pack(side=tk.RIGHT, padx=5)

        # --- 2. Main Display Area (Side-by-Side) ---
        self.images_frame = tk.Frame(self.root, bg="#f0f2f5")
        self.images_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left Panel (Input)
        self.input_frame = tk.Frame(self.images_frame, bg="#ffffff", bd=1, relief="ridge")
        self.input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.input_title = tk.Label(self.input_frame, text=self.LANG_DATA[self.lang]["input_title"], font=title_font, bg="#ffffff", fg="#333")
        self.input_title.pack(pady=10)
        
        self.image_label = tk.Label(self.input_frame, text=self.LANG_DATA[self.lang]["image_label"], bg="#ffffff", font=("Helvetica", 11))
        self.image_label.pack(expand=True)

        # Right Panel (Output)
        self.output_frame = tk.Frame(self.images_frame, bg="#ffffff", bd=1, relief="ridge")
        self.output_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        self.output_title = tk.Label(self.output_frame, text=self.LANG_DATA[self.lang]["output_title"], font=title_font, bg="#ffffff", fg="#333")
        self.output_title.pack(pady=10)
        
        self.output_label = tk.Label(self.output_frame, text=self.LANG_DATA[self.lang]["output_placeholder"], bg="#ffffff", font=("Helvetica", 11))
        self.output_label.pack(expand=True)

        # Bind events
        self.image_label.bind("<Button-1>", self.on_mouse_down)
        self.image_label.bind("<B1-Motion>", self.on_mouse_drag)
        self.image_label.bind("<ButtonRelease-1>", self.on_mouse_up)

    def toggle_language(self):
        self.lang = 'en' if self.lang == 'ar' else 'ar'
        self.lang_toggle_btn.config(text="العربية" if self.lang == 'en' else "English")
        self.update_interface_language()

    def update_interface_language(self):
        self.upload_btn.config(text=self.LANG_DATA[self.lang]["upload_button"])
        self.extract_btn.config(text=self.LANG_DATA[self.lang]["extract_button"])
        self.input_title.config(text=self.LANG_DATA[self.lang]["input_title"])
        self.output_title.config(text=self.LANG_DATA[self.lang]["output_title"])
        
        if self.is_manual_mode:
            self.mode_toggle_btn.config(text=self.LANG_DATA[self.lang]["auto_mode_btn"])
        else:
             self.mode_toggle_btn.config(text=self.LANG_DATA[self.lang]["manual_mode_btn"])
             
        if self.cv_image_rgb is None:
            self.image_label.config(text=self.LANG_DATA[self.lang]["image_label"])
            self.output_label.config(text=self.LANG_DATA[self.lang]["output_placeholder"])

    def toggle_mode(self):
        self.is_manual_mode = not self.is_manual_mode
        if self.is_manual_mode:
            self.mode_toggle_btn.config(text=self.LANG_DATA[self.lang]["auto_mode_btn"], bg="#ffc107") # Warning yellow for manual
            self.selected_box = None
            self.extract_btn.config(state=tk.DISABLED)
            if self.cv_image_rgb is not None:
                self.display_numpy_image(self.cv_image_rgb.copy())
        else:
            self.mode_toggle_btn.config(text=self.LANG_DATA[self.lang]["manual_mode_btn"], bg="#e4e6eb")
            if self.current_image_path:
                 self.process_and_display_image(self.current_image_path)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="اختر صورة" if self.lang == 'ar' else "Select Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.image_label.config(text=self.LANG_DATA[self.lang]["processing"], image="")
            self.output_label.config(text=self.LANG_DATA[self.lang]["output_placeholder"], image="")
            self.extract_btn.config(state=tk.DISABLED)
            self.root.update()
            
            if not self.is_manual_mode:
                self.process_and_display_image(file_path)
            else:
                cv_image = cv2.imread(file_path)
                self.cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                self.display_numpy_image(self.cv_image_rgb.copy())

    def process_and_display_image(self, path):
        cv_image = cv2.imread(path)
        self.cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB) 
        
        results = self.model(self.cv_image_rgb, conf=0.15) 
        self.detected_boxes = []
        self.selected_box = None 
        
        draw_img = self.cv_image_rgb.copy()
        
        for result in results:
            for box in result.boxes.xyxy: 
                x1, y1, x2, y2 = map(int, box[:4])
                self.detected_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 2) # Thinner green lines

        self.display_numpy_image(draw_img)

    def display_numpy_image(self, img_array):
        img_pil = Image.fromarray(img_array)
        orig_w, orig_h = img_pil.size
        
        display_img = img_pil.copy()
        # Adjusted size for side-by-side layout
        display_img.thumbnail((500, 500)) 
        disp_w, disp_h = display_img.size
        
        self.scale_x = orig_w / disp_w
        self.scale_y = orig_h / disp_h
        
        self.tk_image = ImageTk.PhotoImage(display_img)
        self.image_label.config(image=self.tk_image, text="")

    # --- Mouse Event Handlers ---
    
    def on_mouse_down(self, event):
        if self.cv_image_rgb is None:
            return
            
        orig_x = int(event.x * self.scale_x)
        orig_y = int(event.y * self.scale_y)
        
        if self.is_manual_mode:
            self.start_x = orig_x
            self.start_y = orig_y
            self.selected_box = None
            self.extract_btn.config(state=tk.DISABLED)
        else:
            if not self.detected_boxes:
                return
            clicked_box = None
            for box in self.detected_boxes:
                x1, y1, x2, y2 = box
                if x1 <= orig_x <= x2 and y1 <= orig_y <= y2:
                    clicked_box = box
                    break 
                    
            if clicked_box:
                self.selected_box = clicked_box
                self.highlight_selected_box()
                self.extract_btn.config(state=tk.NORMAL) # Enable button

    def on_mouse_drag(self, event):
        if not self.is_manual_mode or self.start_x is None or self.cv_image_rgb is None:
            return
            
        current_x = int(event.x * self.scale_x)
        current_y = int(event.y * self.scale_y)
        
        img_h, img_w = self.cv_image_rgb.shape[:2]
        current_x = max(0, min(current_x, img_w))
        current_y = max(0, min(current_y, img_h))

        x1 = min(self.start_x, current_x)
        y1 = min(self.start_y, current_y)
        x2 = max(self.start_x, current_x)
        y2 = max(self.start_y, current_y)
        
        self.current_rect = (x1, y1, x2, y2)
        
        draw_img = self.cv_image_rgb.copy()
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), (255, 165, 0), 2)
        self.display_numpy_image(draw_img)

    def on_mouse_up(self, event):
        if not self.is_manual_mode or self.start_x is None or self.current_rect is None:
             return
             
        self.selected_box = self.current_rect
        self.start_x = None
        self.start_y = None
        self.current_rect = None
        
        if self.selected_box:
             draw_img = self.cv_image_rgb.copy()
             x1, y1, x2, y2 = self.selected_box
             cv2.rectangle(draw_img, (x1, y1), (x2, y2), (255, 0, 0), 4)
             self.display_numpy_image(draw_img)
             self.extract_btn.config(state=tk.NORMAL) # Enable button

    def highlight_selected_box(self):
        draw_img = self.cv_image_rgb.copy()
        
        for box in self.detected_boxes:
            x1, y1, x2, y2 = box
            if box == self.selected_box:
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (255, 0, 0), 4)
            else:
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
        self.display_numpy_image(draw_img)

    def extract_and_save(self):
        if self.selected_box is None or self.cv_image_rgb is None:
            return
            
        self.extract_btn.config(text=self.LANG_DATA[self.lang]["extracting"], state=tk.DISABLED)
        self.output_label.config(text=self.LANG_DATA[self.lang]["processing"], image="")
        self.root.update()

        try:
            img = self.cv_image_rgb.copy()
            x1, y1, x2, y2 = self.selected_box

            padding_percent = 0.15
            w = x2 - x1
            h = y2 - y1
            
            pad_x = int(w * padding_percent)
            pad_y = int(h * padding_percent)
            
            img_h, img_w = img.shape[:2]
            px1 = max(0, x1 - pad_x)
            py1 = max(0, y1 - pad_y)
            px2 = min(img_w, x2 + pad_x)
            py2 = min(img_h, y2 + pad_y)

            blurred_img = cv2.GaussianBlur(img, (99, 99), 30)
            sharp_roi = img[py1:py2, px1:px2]

            manipulated_img = blurred_img.copy()
            manipulated_img[py1:py2, px1:px2] = sharp_roi

            manipulated_pil = Image.fromarray(manipulated_img)
            
            # Run rembg
            final_output = remove(manipulated_pil)
            
            # 1. Save the result
            final_output.save("isolated_object.png")
            
            # 2. Display the result in the output panel
            display_out = final_output.copy()
            display_out.thumbnail((500, 500))
            self.tk_output_image = ImageTk.PhotoImage(display_out)
            self.output_label.config(image=self.tk_output_image, text="")

            messagebox.showinfo("Success", self.LANG_DATA[self.lang]["success"])
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            
        finally:
            self.extract_btn.config(text=self.LANG_DATA[self.lang]["extract_button"], state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectExtractorApp(root)
    root.mainloop()