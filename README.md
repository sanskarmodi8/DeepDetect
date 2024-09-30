# Project DeepDetect

<br/>

In the age of deepfakes, seeing is no longer believing. But even if you can't trust your eyes, you can trust us.
<br/>

[DeepDetect](https://huggingface.co/spaces/SanskarModi/DeepDetect) is an AI Powered Application to help you see the reality.
<br/>
<br/>
[![DeepDetect App](https://github.com/sanskarmodi8/DeepDetect/blob/main/.github_assets/app.png?raw=true)](https://huggingface.co/spaces/SanskarModi/DeepDetect)


<br/>

## ⚠️ Important Notice

**Please Note**: Due to limited computational resources, this model is currently trained on a relatively small dataset with a short `sequence_length` of 10. As a result, its performance metrics are as follows:
- Accuracy: ~76%
- F1 Score: ~76%
- AUC: ~85%

While we are confident the results wil be promising after traning the model on a greater sequence length when we get access to additional computational resources, we strongly advise against using this project for serious or commercial applications in its current state. It serves best as a proof-of-concept and educational purposes.

## 🚀 Features

- AI-powered deepfake detection
- User-friendly Gradio interface
- Comprehensive ML pipeline with MLflow and DVC

## 🛠️ Tech Stack

- **Data Manipulation**: NumPy, Pandas
- **Model**: Scikit-learn, PyTorch
- **Image and Video processing**: OpenCV, MTCNN, face_recognition
- **Visualization**: Plotly
- **MLOps**: MLflow, DVC
- **UI**: Gradio
- **Deployment**: HuggingFace Spaces

## 🏗️ Project Structure

```
DeepDetect/
│
├── src/                            # Source code directory
│   └── DeepfakeDetection/          # Main package for the deepfake detection functionality
│       ├── components/             # Modular components of the detection pipeline (e.g., data loading, preprocessing, model training)
│       ├── utils/                  # Utility functions and helper modules
│       ├── config/                 # Configuration file for different components
│       ├── pipeline/               # Scripts defining the overall detection pipeline
│       ├── entity/                 # Data entity classes
│       └── constants/              # Constant values used throughout the project
│
├── config/                         # Global configuration files
├── app.py                          # Main application file using Gradio
├── Dockerfile                      # Instructions for building a Docker container
├── requirements.txt                # Python dependencies for the project
├── pyproject.toml                  # Project metadata and build system requirements
├── main.py                         # Entry point for running the entire pipeline
├── params.yaml                     # Parameters for model training and evaluation
├── dvc.yaml                        # Data Version Control configuration
├── structure.py                    # Script to generate or manage project structure
└── format.sh                       # Shell script for code formatting
```

## 🚀 Setup

1. Clone this repository
2. Install Python >=3.9
3. Run: `pip install -r requirements.txt`
4. Install Docker (optional)
5. Install CMake: `sudo apt install cmake` (Ubuntu) or download from [CMake Official Website](https://cmake.org/download/)
6. Download the [FaceForensics++](https://github.com/ondyari/FaceForensics) data. Make sure to replace the value of `source_data` in `config/config.yaml` with the correct path to the downloaded data folder on your system.

## 🖥️ Usage

### Quick Start

```bash
dvc init --force
dvc repro
```

or

```bash
python main.py
```

### Detailed Pipeline Execution

```bash
# Data Ingestion
python src/DeepfakeDetection/pipeline/stage_01_data_ingestion.py

# Data Preprocessing
python src/DeepfakeDetection/pipeline/stage_02_data_preprocessing.py

# Model Training
python src/DeepfakeDetection/pipeline/stage_03_model_training.py

# Model Evaluation
python src/DeepfakeDetection/pipeline/stage_04_model_evaluation.py
```

If you intend to use MLFLOW then you have to save ytour MLFLOW credentials in .env file at root level and update `src/DeepDetect/pipeline/stage_03_model_training.py` and `src/DeepDetect/pipeline/stage_04_model_evaluation.py` accordingly as mentioned in the comments.

### Launch Gradio Interface

```bash
python app.py
```

## 🔮 Future Goals

- [ ] Use Attention weights along with Feature maps in Grad-CAM for enhanced explainability
- [ ] Use MTCNN in Preprocessing component instead of face_recognition
- [ ] Scale up training with larger datasets and sequence lengths
- [ ] Add user-configurable sequence length in the application

## 🤝 Contributing

We welcome contributions! Here's how:

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -am 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the [GPL-3.0 License](https://www.gnu.org/licenses/gpl-3.0.en.html).

---

<p align="center">
  Made with ❤️ by Sanskar Modi
</p>
