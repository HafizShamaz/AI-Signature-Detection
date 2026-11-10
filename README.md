# AI Signature Detection System

An AI-powered signature detection and comparison system built with **Deep Learning, Computer Vision, FastAPI, Docker, and Microsoft Azure**.

The project is designed to detect handwritten signatures in document images, generate signature similarity scores, and provide API-based access to the inference services. It also includes a foundation for cloud deployment, model evaluation, and MLOps automation.

## Project Overview

The AI Signature Detection System combines two computer vision capabilities:

1. **Signature Detection:** Identifies and localizes handwritten signatures in document images using an object detection model.
2. **Signature Similarity Comparison:** Compares two signature images using a Siamese neural network and calculates a similarity score.

The system exposes these capabilities through a REST API and is structured to support containerized deployment and cloud-based infrastructure.

## Key Features

- Detect handwritten signatures in document images.
- Locate signatures using bounding-box coordinates.
- Generate confidence scores for detected signatures.
- Compare two signature images.
- Calculate cosine-based signature similarity scores.
- Classify comparisons as similar or different using a configurable threshold.
- Upload and process supported document and image formats.
- REST API built with FastAPI.
- Background processing architecture with a worker service.
- Local storage and Azure Blob Storage integration.
- Docker-based application packaging.
- Automated testing through GitHub Actions.
- Azure cloud infrastructure for deployment and MLOps.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python 3.11 |
| Deep Learning | PyTorch |
| Computer Vision | Torchvision, Pillow |
| Object Detection | Faster R-CNN |
| Detection Backbone | MobileNetV3 Large FPN |
| Signature Comparison | Siamese Neural Network |
| Feature Extraction | ResNet18 |
| Similarity Measurement | Cosine Similarity |
| API Framework | FastAPI |
| API Server | Uvicorn |
| Data Validation | Pydantic |
| Database | SQLite |
| Cloud Storage | Azure Blob Storage |
| Cloud Infrastructure | Microsoft Azure |
| Containerization | Docker, Docker Compose |
| Container Registry | Azure Container Registry |
| Machine Learning Platform | Azure Machine Learning |
| Model Tracking | MLflow |
| Monitoring | Azure Application Insights |
| Secrets Management | Azure Key Vault |
| CI/CD | GitHub Actions |
| Testing | Pytest |
| Version Control | Git and GitHub |

## System Architecture

```text
                 USER / CLIENT
                       |
                       v
                FASTAPI REST API
                       |
             Document Upload Service
                       |
                       v
              Job / Request Handling
                       |
              +--------+--------+
              |                 |
              v                 v
        Background Worker   API Services
              |                 |
              v                 v
       Signature Detection  Signature Comparison
        Faster R-CNN        Siamese Network
        MobileNetV3         ResNet18 Backbone
              |                 |
              +--------+--------+
                       |
                       v
                Prediction Results
                       |
             +---------+---------+
             |                   |
             v                   v
       Local / Azure         SQLite Job
       Blob Storage          Repository
```

## Machine Learning Models

### 1. Signature Detection Model

The object detection component uses **Faster R-CNN with a MobileNetV3 Large FPN backbone**.

Its purpose is to identify signature regions within document images.

**Model configuration:**

- Architecture: Faster R-CNN
- Backbone: MobileNetV3 Large FPN
- Number of classes: 2 (background and signature)
- Output: Bounding boxes and confidence scores
- Framework: PyTorch and Torchvision

**Evaluation results**

The following results were obtained from the project's test dataset:

| Metric | Result |
|---|---:|
| Precision | 89.74% |
| Recall | 91.94% |
| F1 Score | 90.82% |
| mAP@0.5 | 94.37% |
| mAP@0.5:0.95 | 62.28% |
| Average inference time (CPU) | 0.3194 seconds |
| CPU inference speed | 3.13 FPS |

These are measured results from the current evaluated model and may vary with hardware, image resolution, and inference configuration.

### 2. Signature Similarity Model

The signature comparison component uses a **Siamese Neural Network** with a ResNet18 feature extraction backbone.

It generates 128-dimensional feature embeddings for signature images and compares them using cosine similarity.

**Model configuration:**

- Architecture: Siamese Neural Network
- Feature extractor: ResNet18
- Embedding dimension: 128
- Image size: 128 × 192
- Similarity metric: Cosine similarity
- Current comparison threshold: Approximately 0.718

The comparison service returns:

- Similarity score
- Configured threshold
- Comparison decision (`similar` or `different`)

**Important:** Similarity indicates visual resemblance between two signatures. It does not independently establish signer identity, document authenticity, or forgery detection accuracy.

## Supported File Formats

The document upload API supports:

- PDF
- PNG
- JPG / JPEG
- TIFF
- BMP

Maximum configured upload size: 25 MB.

## Project Structure

```text
AI-Signature-Detection/
│
├── app/
│   ├── main.py
│   ├── config.py
│   │
│   ├── routes/
│   │   ├── jobs.py
│   │   ├── predictions.py
│   │   └── upload.py
│   │
│   ├── services/
│   │   ├── inference.py
│   │   ├── repository.py
│   │   └── storage.py
│   │
│   └── schemas/
│
├── models/
│   ├── signature_detector.pt
│   └── signature_similarity.pt
│
├── similarity/
│   ├── __init__.py
│   └── model.py
│
├── worker/
│   ├── main.py
│   └── processor.py
│
├── docker/
│   ├── api.Dockerfile
│   └── worker.Dockerfile
│
├── mlops/
│
├── training/
│   └── evaluate.py
│
├── tests/
│
├── reports/
│
├── requirements.txt
├── requirements-api.txt
├── docker-compose.yml
└── README.md
```

The training dataset is maintained separately from the source repository to avoid committing large dataset files.

## Installation and Local Setup

### Prerequisites

Install the following:

- Python 3.11
- Git
- Docker Desktop
- GitHub account (for repository access)

### 1. Clone the Repository

```bash
git clone https://github.com/HafizShamaz/AI-Signature-Detection.git

cd AI-Signature-Detection
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements-api.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root.

Configure the environment according to the settings defined in `app/config.py`.

Do not commit `.env` files, credentials, API keys, or connection strings to GitHub.

### 5. Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Access API Documentation

Once the API is running:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health endpoint: http://localhost:8000/health

## Running with Docker

Docker Compose provides a containerized API and background worker.

Build the application:

```bash
docker compose build
```

Start the services:

```bash
docker compose up -d
```

Check running containers:

```bash
docker compose ps
```

Check API health:

```bash
curl http://localhost:8000/health
```

Stop the services:

```bash
docker compose down
```

The application uses a shared Docker volume for local application data and model files mounted into the relevant containers.

## API Capabilities

The API provides endpoints for:

- Health monitoring.
- Document and image uploads.
- Job submission and status tracking.
- Prediction result retrieval.
- Signature similarity comparison.

Explore the complete endpoint definitions and request schemas through the Swagger documentation at `/docs`.

## Dataset

The project uses a signature detection dataset organized into training, validation, and testing partitions.

| Dataset Split | Images | Signature Instances |
|---|---:|---:|
| Training | 22,953 | 29,952 |
| Validation | 3,287 | 4,670 |
| Testing | 2,718 | 3,472 |
| **Total** | **28,958** | **38,094** |

The dataset uses YOLO-style annotation files and includes the following dataset configuration:

```yaml
path: .
train: train/images
val: valid/images
test: test/images

nc: 1
names:
  - signature
```

The dataset is not included as part of the standard source-code checkout. It is maintained separately and can be stored in Azure Blob Storage.

## Cloud Infrastructure – Microsoft Azure

The project is being prepared for deployment using Microsoft Azure services.

| Azure Service | Purpose |
|---|---|
| Azure Resource Group | Centralized resource management |
| Azure Blob Storage | Dataset, document, and model-artifact storage |
| Azure Container Registry | Docker image registry |
| Azure Container Apps | Containerized application hosting |
| Azure Machine Learning | Machine learning workspace |
| Azure Key Vault | Secrets management |
| Azure Application Insights | Application monitoring |
| Azure Log Analytics | Container and infrastructure logs |
| MLflow | Experiment and model tracking |

Cloud resources are provisioned separately from the local development environment.

## CI/CD – GitHub Actions

GitHub Actions is used to automate repository validation.

The CI workflow is configured to:

1. Trigger on pushes to the `main` branch.
2. Trigger on pull requests targeting `main`.
3. Check out the repository.
4. Set up Python 3.11.
5. Install application dependencies and Pytest.
6. Execute the automated test suite.

Workflow configuration:

```text
.github/
└── workflows/
    └── ci-cd.yml
```

The current workflow focuses on automated validation. Automated cloud deployment and model promotion can be added as subsequent pipeline stages.

## Testing

Run the automated tests locally:

```bash
python -m pytest tests/ -v
```

The test suite covers application functionality and available inference components.

Some integration tests require model checkpoints, sample images, or the full dataset. Tests that depend on unavailable local resources may be skipped in CI.

## Security Considerations

- Store credentials in environment variables or Azure Key Vault.
- Never commit `.env` files or cloud access keys.
- Validate uploaded file types and sizes.
- Restrict access to stored documents and model artifacts.
- Use appropriate authentication and authorization before production exposure.
- Monitor application errors and unusual activity.
- Treat similarity scores as decision-support information, not definitive identity verification.




---

*Built with Python, PyTorch, FastAPI, Docker, and Microsoft Azure.*
