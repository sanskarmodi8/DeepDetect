# DeepDetect

This project uses a small part of [Deepfake Detection Challenge Dataset](https://www.kaggle.com/c/deepfake-detection-challenge/data) from Kaggle and aims to classify a video as deepfake or not.

Deployment is done on Hugging Faces using Gradio, [click here]() to visit the deployed app.

![image](https://imgs.search.brave.com/Ix_KaGH7ekl6m8g5ww7CMc177tacKIBzNlJfVC-VnEQ/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9zcGVj/dHJ1bS5pZWVlLm9y/Zy9tZWRpYS1saWJy/YXJ5L3N0aWxsLWZy/b20tYS12aWRlby1z/aG93aW5nLWEtd29t/YW4taW4tYS1raXRj/aGVuLXRoZS1vcmln/aW5hbC1pcy1vbi10/aGUtbGVmdC10aGUt/ZGVlcGZha2Utb24t/dGhlLXJpZ2h0LWFs/b25nLXdpdGgtYW4t/aW5zZXQtaW1hLmpw/Zz9pZD0yNTU5MDM4/MyZ3aWR0aD0xMjAw/JmhlaWdodD02NjA)

This project mainly utilizes following tools and libraries :

- Numpy, Pandas, Scikit Learn, Keras, VIT Keras, Tensorflow (for data preproccessing and model building)
- MLFLOW and Dagshub (for experiment tracking and model registry)
- DVC (for pipeline versioning)
- Gradio (for user interface)
- Hugging Faces (for deployment)
- Docker (for containerization)

## Table of Contents

- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

The project follows a modular structure for better organization and maintainability. Here's an overview of the directory structure:

- `.github/workflows`: GitHub Actions workflows for CI/CD.
- `src/`: Source code directory.
  - `DeepfakeDetection/`
    - `components/`: Modules for different stages of the pipeline.
    - `utils/`: Utility functions.
    - `config/`: Configuration for each of the components.
    - `pipeline/`: Scripts for pipeline stages.
    - `entity/`: Data entity classes.
    - `constants/`: Constants used throughout the project.
- `config/`: Base Configuration for each stage of the project.
- `notebook/`: Directory for trials, experiments and prototype code in jupyter notebook.
- `gradio_app.py`: User interface using gradio.
- `Dockerfile`: Docker configuration for containerization.
- `requirements.txt`: Project dependencies.
- `pyproject.toml`: Build system configuration for project.
- `main.py`: Main script for execution of the complete pipeline.
- `params.yaml`: All the parameters used in the complete pipeline.
- `dvc.yaml`: Configuration file for DVC Pipeline Versioning.

## Setup

To set up the project environment, follow these steps:

1. Clone this repository.
2. Install Python 3.8 and ensure pip is installed.
3. Install project dependencies using `pip install -r requirements.txt`.
4. Ensure Docker is installed if you intend to use containerization.

## Usage

### To directly run the complete Data ingestion, Data cleaning, Model preparation and training and Model evaluation pipeline

run the command

```bash
dvc init
dvc repro
```

### To explicitly run each pipeline follow following commands-

#### Data Ingestion

To download and save the dataset, run:

```bash
python src/DeepfakeDetection/pipeline/stage_01_data_ingestion.py
```

#### Preprocessing the Data

To preprocess and save the cleaned data, run:

```bash
python src/DeepfakeDetection/pipeline/stage_02_data_preprocessing.py
```

#### Model Preparation and Training

To train the model, execute:

```bash
python src/DeepfakeDetection/pipeline/stage_03_model_training.py
```

#### Model Evaluation

To evaluate the trained model, run:

```bash
python src/DeepfakeDetection/pipeline/stage_04_model_evaluation.py
```

### To start the gradio application :

```bash
python gradio_app.py
```

## Future Plans

 - Providing explainibility and confidence heat maps
 - Implementing Feedback Mechanism
 - Real time Inferencing
 - Optimization and quantization
 - Add extra preprocessing steps

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/new-feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Create a new Pull Request.

Please ensure that your contributions adhere to the project's coding standards.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
