# Project DeepDetect

<br/>

In an era where seeing is no longer believing, [DeepDetect](https://huggingface.co/spaces/SanskarModi/DeepDetect) empowers you to unveil the truth behind digital content.

[![DeepDetect App](https://github.com/sanskarmodi8/DeepDetect/blob/main/.github_assets/app.png?raw=true)](https://huggingface.co/spaces/SanskarModi/DeepDetect)

[Try DeepDetect Now](https://huggingface.co/spaces/SanskarModi/DeepDetect)

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
├── src/
│   └── DeepfakeDetection/
│       ├── components/
│       ├── utils/
│       ├── config/
│       ├── pipeline/
│       ├── entity/
│       └── constants/
│
├── config/
├── app.py
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── main.py
├── params.yaml
├── dvc.yaml
├── structure.py
└── format.sh
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

### Launch Gradio Interface

```bash
python app.py
```

## 🔮 Future Goals

- [ ] Implement Grad-CAM for enhanced explainability
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
  Made with ❤️ by [Sanskar Modi]
</p>
