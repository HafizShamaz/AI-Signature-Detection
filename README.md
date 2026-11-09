# AI-Powered Document Signature Detection System

An end-to-end AI-powered document signature detection system built using **YOLO11s (You Only Look Once)** to automatically identify, locate, and analyze handwritten signatures in digital documents.

The system supports multiple document formats, including PNG, JPG, JPEG, TIFF, BMP, and TXT and uses computer vision to detect signatures within document images. It provides the total number of detected signatures along with individual confidence scores and bounding-box locations, enabling efficient and accurate document verification workflows.

The project incorporates a complete machine learning workflow, including synthetic dataset generation, strict dataset preprocessing, YOLO-format annotation handling, model training, evaluation, and deployment.

### Key Features

- **Multi-Format Document Processing:** Supports various document formats, including PDF, images, and text-based documents.
- **Signature Detection:** Uses the YOLO11s object detection model to identify handwritten signatures.
- **Detection Analytics:** Reports the total number of detected signatures and individual confidence scores.
- **Bounding Box Localization:** Highlights the exact location of each detected signature.
- **Synthetic Dataset Generation:** Generates realistic synthetic document datasets for model training and experimentation.
- **Dataset Preprocessing:** Removes duplicate or highly similar images, documents without signatures, and images with missing annotation files.
- **Model Training & Evaluation:** Implements a YOLO-based training and evaluation pipeline.
- **Web-Based Interface:** Allows users to upload documents and receive signature detection results.
- **REST API:** Integrates FastAPI for document upload, processing, and prediction.
- **Containerized Deployment:** Uses Docker and Docker Compose for consistent deployment.
- **MLOps Integration:** Designed for automated model training, experiment tracking, deployment, and continuous model improvement.

### Technology Stack

- **Programming Language:** Python
- **Deep Learning:** PyTorch
- **Object Detection:** YOLO11s
- **Computer Vision:** OpenCV
- **API Framework:** FastAPI
- **Containerization:** Docker, Docker Compose
- **Cloud & MLOps:** Microsoft Azure Machine Learning
- **Dataset Annotation:** YOLO Bounding Box Format
- **Development Environment:** Google Colab, Visual Studio Code

### Project Objective

The primary objective is to develop an intelligent document analysis solution that automates signature identification and localization, reducing the need for manual document inspection. The system is designed to provide a scalable foundation for document verification, digital workflows, and automated document processing.

**Project Focus:** Computer Vision | Deep Learning | Object Detection | Document AI | MLOps | Model Deployment
