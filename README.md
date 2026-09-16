<p align="center">
  <img src="RP_background_remover.ico" alt="Centered image" width="300">
</p>


# Smart Object Extractor ✂️

A lightweight, intelligent desktop application built with Python and Tkinter for extracting specific objects from images with high-precision edges. The tool combines the semantic understanding of **YOLOv8** with the pixel-perfect alpha matting of **rembg** (U²-Net) using a custom Saliency Manipulation pipeline.

## ✨ Features

- **🤖 Auto-Detection Mode:** Automatically detects common objects in the image using YOLOv8, allowing you to extract them with a single click.
- **🖱️ Manual Selection Mode:** A fallback mode that lets you manually draw a bounding box around any custom object that YOLO might miss.
- **🪄 Saliency Manipulation Pipeline:** Instead of hard-cropping, the app applies a heavy Gaussian blur to the background before passing it to `rembg`. This preserves the spatial context and ensures flawless extraction of extended edges (like hair or fur).
- **🌍 Bilingual Interface:** Seamlessly switch between English and Arabic UI.
- **🚀 Smart Initialization:** Features a splash screen that handles the automatic downloading of AI model weights (YOLO and U²-Net) smoothly during the first run.

## 🛠️ Tech Stack & Libraries

- **[Tkinter](https://docs.python.org/3/library/tkinter.html):** For the Graphical User Interface (GUI).
- **[OpenCV (cv2)](https://opencv.org/):** For image processing, bounding box drawing, and applying the Gaussian blur filter.
- **[Ultralytics (YOLOv8)](https://github.com/ultralytics/ultralytics):** For fast and lightweight object detection (`yolov8n.pt`).
- **[Rembg](https://github.com/danielgatis/rembg):** For deep-learning-based background removal utilizing the U²-Net architecture.
- **[Pillow (PIL)](https://python-pillow.org/):** For managing image rendering within the Tkinter canvas and handling transparent PNGs.

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.8 or higher installed on your system. 

### 2. Install Dependencies
Open your terminal or command prompt and run the following command to install the required libraries. *(Note: We use the `[cpu]` flag for `rembg` to ensure compatibility across standard machines without requiring dedicated GPUs).*

```bash
pip install ultralytics opencv-python pillow rembg[cpu]
```

> <p> <b>Note on First Run:</b> Upon launching the app for the very first time, a setup splash screen will appear. The app will download the necessary AI models (<code>yolov8n.pt</code> and <code>U²-Net</code> weights) in the background. This requires an internet connection and may take a few minutes. Subsequent launches will take only seconds and can work completely offline.</p>

## 🧠 How the Pipeline Works (Under the Hood)

1. **Object Localization:** The user selects an object either via YOLO's auto-generated bounding boxes or by drawing a manual rectangle.
2. **Safety Padding:** The app programmatically expands the selected bounding box by 15% to ensure extended parts of the object are not cut off.
3. **Background Blurring:** OpenCV applies a strong Gaussian Blur (`99x99` kernel) to the entire image.
4. **Sharp ROI Replacement:** The original sharp pixels of the selected object's padded bounding box are pasted back on top of the blurred background.
5. **Alpha Matting:** The modified image is fed to `rembg`. Because the background lacks high-frequency details (due to the blur), the U²-Net model is forced to focus its attention entirely on the sharp object, resulting in a perfect, clean extraction.

