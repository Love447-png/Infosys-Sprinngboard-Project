Project name: “CliniScan: Lung-Abnormality Detection on Chest Xrays using AI”
Mentor: G.K.S Jyoteesh
Project Description
To develop an AI-powered system that can automatically detect and localize lung
abnormalities from chest X-ray images using deep learning techniques. The system
aims to assist radiologists and healthcare providers by identifying key pathological
findings such as opacities, consolidations, fibrosis, and masses, and optionally
classifying related pulmonary conditions like pneumonia or tuberculosis. The solution
will be trained on the VinDr-CXR dataset and optimized for clinical relevance,
interpretability, and deployment readiness in real-world diagnostic settings. Project
Workflow: Lung-Abnormality Detection on Chest X-Rays
Project Workflow
1. Data Acquisition
Download the VinDr-CXR dataset (18,000 CXR images with bounding boxes and
diagnosis labels).
https://physionet.org/content/vindr-cxr/1.0.0/
2. Preprocessing
Convert DICOM to PNG/JPEG, resize images, normalize, and parse annotations for
model input.
3. Model Development
Classification: Detect multiple findings per image using CNNs (e.g., ResNet,
EfficientNet).
Detection: Localize abnormalities using object detection models (e.g., YOLOv8, Faster
R-CNN).
4. Training &amp; Evaluation
Train on labeled data with augmentation. Evaluate using AUC, F1-score, or mAP
depending on task.
5. Visualization &amp; Interpretation
Use Grad-CAM or bounding boxes to visualize results for clinical interpretability. 6.
Deployment (Optional)
Deploy as a web-based tool or integrate into a clinical viewer for decision support.
Tech Stack
● Data &amp; Annotation

○ Dataset: VinDr-CXR
○ Format: DICOM images, CSV annotations (bounding boxes and labels)
○ Tools: pydicom, pandas, OpenCV
● Preprocessing
○ Image handling: pydicom, Pillow, OpenCV
○ Data processing: pandas, NumPy
○ Annotation conversion: custom Python scripts (to YOLO or COCO format)
● Model Development
○ Frameworks: PyTorch or TensorFlow
○ Models:
For classification: EfficientNet, ResNet, DenseNet
For detection: YOLOv8 (via Ultralytics), Faster R-CNN (via torchvision)
● Training and Evaluation
○ Training libraries: PyTorch Lightning or Keras
○ Evaluation metrics: scikit-learn (AUC, F1-score), torchmetrics, mean Average
Precision (mAP) for detection
● Visualization
○ For heatmaps: Grad-CAM, matplotlib, seaborn
○ For bounding boxes: OpenCV, matplotlib, Albumentations (for augmentation) ●
Deployment (Optional)
○ Web interface: Streamlit, Flask, or Gradio
○ Model serving: ONNX, TorchScript, or TensorFlow SavedModel
