<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="BioDCASE_header_light.svg">
    <source media="(prefers-color-scheme: light)" srcset="BioDCASE_header_dark.svg">
    <img src="BioDCASE_header_dark.svg" alt="BioDCASE Logo" width="600">
  </picture>
  
  <br><br>
</div>

This repository contains the development framework for the **BioDCASE-Tiny 2025 competition (Task 3)**, focusing on TinyML implementation for bird species recognition on the ESP32-S3-Korvo-2 development board.

For complete competition details, visit the [official BioDCASE 2025 Task 3 website](https://biodcase.github.io/challenge2025/task3).

## Background

BioDCASE-Tiny is a competition for developing efficient machine learning models for bird audio recognition that can run on resource-constrained embedded devices. The project uses the ESP32-S3-Korvo-2 development board, which offers audio processing capabilities in a small form factor suitable for field deployment.

## Table of Contents
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Development](#development)
  - [Data Processing Pipeline](#data-processing-pipeline)
  - [Model Training](#model-training)
  - [ESP32-S3 Deployment](#esp32-s3-deployment)
- [ESP32-S3-Korvo-2 Development Board](#esp32-s3-korvo-2-development-board)
- [Code Structure](#code-structure)
- [Development Tips](#development-tips)
- [Evaluation Metrics](#evaluation-metrics)
- [License](#license)
- [Citation](#citation)
- [Funding](#funding)
- [Partners](#partners)

## Setup and Installation

### Prerequisites

1. Python 3.11+ with pip and venv
2. [Docker](https://www.docker.com/get-started) for ESP-IDF environment
3. USB cable and ESP32-S3-Korvo-2 development board

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/birdnet-team/BioDCASE-Tiny-2025.git
cd BioDCASE-Tiny-2025
```

2. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install Python dependencies:
```bash
pip install -e .
```

4. Set your serial device port in the pipeline_config.yaml

```yaml
embedded_code_generation:
  serial_device: <YOUR_DEVICE> 
```

### Running on Windows

As the required tflite-micro package is not easily available for Windows we recommend using WSL to run this project.

To make your device accessible for WSL you can use this guide: https://learn.microsoft.com/en-us/windows/wsl/connect-usb 

To determine your serial device port you can use the following command:

```bash
dmesg | grep tty
```

You might also need to grant some rights to run the deployment:

```bash
sudo adduser $USER dialout
sudo chmod a+rw $SERIAL_PORT
```

## Usage
- Modify model.py with your architecture (make sure to compile with optimizer and loss)
- Modify the training loop in the same file, if you need to
- Modify pipeline_config.yaml parameters of feature extraction
- run biodcase.py

## Dataset

The BioDCASE-Tiny 2025 competition uses a specialized dataset of Yellowhammer bird vocalizations. Key features include:

- 2+ hours of audio recordings
- Songs from multiple individuals recorded at various distances (6.5m to 200m) 
- Recordings in different environments (forest and grassland)
- Includes negative samples (other bird species and background noise)
- Dataset is split into training, validation, and evaluation sets

The training set contains recordings from 8 individuals, while the validation set contains recordings from 2 individuals. An additional 2 individuals are reserved for the final evaluation.

### Dataset Structure

The dataset is organized as follows:

```
Development_Set/
├── Training_Set/
│   ├── Yellowhammer/
│   │   └── *.wav (Yellowhammer vocalizations)
│   └── Negatives/
│       └── *.wav (Other species and background noise)
└── Validation_Set/
    ├── Yellowhammer/
    │   └── *.wav (Yellowhammer vocalizations)
    └── Negatives/
        └── *.wav (Other species and background noise)
```

- **Yellowhammer/** - Contains target species vocalizations with filenames following format `YH_SongID_Location_Distance.wav`
  - SongID: 3-digit identifier for each song
  - Location: "forest", "grassland", or "speaker" for original recordings
  - Distance: A (original) through H (farthest at ~200m)

- **Negatives/** - Contains negative samples with filenames following format `Type_ID.wav`
  - Type: "Background" for noise or "bird" for non-target species vocalizations
  - ID: File identifier

### Download

Download the dataset from: [BioDCASE-Tiny 2025 Dataset](https://doi.org/10.5281/zenodo.15228365)

After downloading paste the folders into /data/01_raw/clips

## Development

### Quickstart

To run the complete pipeline execute:
   ```bash
   python biodcase.py
   ```

This will execute the data preprocessing, extract the features, train the model and deploy it to your board.

Once deployed, benchmarking code on the ESP32-S3 will display info, via serial monitor, about the runtime performance of the preprocessing steps and actual model.

#### Step-by-Step Deployment Instructions

The steps of the pipeline can be executed individually

1. Data Preprocessing
   ```bash
   python data_preprocessing.py
   ```

2. Feature Extraction
   ```bash
   python feature_extraction.py
   ```

3. Model Training
   ```bash
   python model_training.py
   ```

4. Deployment
   ```bash
   python embedded_code_generation.py
   ```


### Data Processing Pipeline

The data processing pipeline follows these steps:
1. Raw audio files are read and preprocessed
2. Features are extracted according to configuration in `pipeline_config.yaml`
3. The dataset is split into training/validation/testing sets
4. Features are used for model training

### Model Training

The model training process is managed in `model_training.py`. You can customize:
- Model architecture in `model.py` and, optionally, the training loop
- Training hyperparameters in `pipeline_config.yaml`
- Feature extraction parameters to optimize model input

### ESP32-S3 Deployment

To deploy your model to the ESP32-S3-Korvo-2 board, you'll use the built-in deployment tools that handle model conversion, code generation, and flashing. The deployment process:

1. Converts your trained Keras model to TensorFlow Lite format optimized for the ESP32-S3
2. Packages your feature extraction configuration for embedded use
3. Generates C++ code that integrates with the ESP-IDF framework
4. Compiles the firmware using Docker-based ESP-IDF toolchain
5. Flashes the compiled firmware to your connected ESP32-S3-Korvo-2 board

## ESP32-S3-Korvo-2 Development Board

The [ESP32-S3-Korvo-2](https://www.digikey.de/de/products/detail/espressif-systems/ESP32-S3-KORVO-2/15822448) board features:
- ESP32-S3 dual-core processor
- Built-in microphone array
- Audio codec for high-quality audio processing
- Wi-Fi and Bluetooth connectivity
- USB-C connection for programming and debugging

## Code Structure

### Key Entry Points

- `biodcase.py` - Main execution pipeline
- `model.py` - Define your model architecture
- `feature_extraction.py` - Audio feature extraction implementations
- `embedded_code_generation.py` - ESP32 code generation utilities
- `biodcase_tiny/embedded/esp_target.py` - ESP target definition and configuration
- `biodcase_tiny/embedded/firmware/main` - Firmware source code

### Benchmarking

The codebase includes performance benchmarking tools that measure:
- Feature extraction time
- Model inference time
- Memory usage on the target device

## Development Tips

1. **Feature Extraction Parameters**: Carefully tune the feature extraction parameters in `pipeline_config.yaml` for your specific audio dataset.

2. **Model Size**: Keep your model compact. The ESP32-S3 has limited memory, so optimize your architecture accordingly.

3. **Profiling**: Use the profiling tools to identify bottlenecks in your implementation.

4. **Memory Management**: Be mindful of memory allocation on the ESP32. Monitor the allocations reported by the firmware.

5. **Docker Environment**: The toolchain uses Docker to provide a consistent ESP-IDF environment, making it easier to build on any host system.

## Evaluation Metrics

The BioDCASE-Tiny competition evaluates models based on multiple criteria:

### Classification Performance
- **Average precision**: the average value of precision across all recall levels from 0 to 1. 

### Resource Efficiency
- **Model Size**: Tflite model file size (KB)
- **Inference Time**: Average time required for single audio classification, including feature extraction (ms)
- **Peak Memory Usage**: Maximum RAM usage during inference (KB)

### Ranking
Participants will be ranked separately for each one of the evaluation criteria.

## License

This project is licensed under the Apache License 2.0 - see the license headers in individual files for details.

## Citation

If you use the BioDCASE-Tiny framework or dataset in your research, please cite the following:

### Framework Citation

```bibtex
@misc{biodcase_tiny_2025,
  author = {Carmantini, Giovanni and Förstner, Friedrich and Isik, Can and Kahl, Stefan},
  title = {BioDCASE-Tiny 2025: A Framework for Bird Species Recognition on Resource-Constrained Hardware},
  year = {2025},
  institution = {Cornell University and Chemnitz University of Technology},
  type = {Software},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/birdnet-team/BioDCASE-Tiny-2025}}
}
```

### Dataset Citation

```bibtex
@dataset{yellowhammer_dataset_2025,
  author = {Morandi, Ilaria and Linhart, Pavel and Kwak, Minkyung and Petrusková, Tereza},
  title = {BioDCASE 2025 Task 3: Bioacoustics for Tiny Hardware Development Set},
  year = {2025},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.15228365},
  url = {https://doi.org/10.5281/zenodo.15228365}
}
```

## Funding

This project is supported by Jake Holshuh (Cornell class of ´69) and The Arthur Vining Davis Foundations.
Our work in the K. Lisa Yang Center for Conservation Bioacoustics is made possible by the generosity of K. Lisa Yang to advance innovative conservation technologies to inspire and inform the conservation of wildlife and habitats.

The development of BirdNET is supported by the German Federal Ministry of Education and Research through the project “BirdNET+” (FKZ 01|S22072).
The German Federal Ministry for the Environment, Nature Conservation and Nuclear Safety contributes through the “DeepBirdDetect” project (FKZ 67KI31040E).
In addition, the Deutsche Bundesstiftung Umwelt supports BirdNET through the project “RangerSound” (project 39263/01).

## Partners

BirdNET is a joint effort of partners from academia and industry.
Without these partnerships, this project would not have been possible.
Thank you!

![Logos of all partners](https://tuc.cloud/index.php/s/KSdWfX5CnSRpRgQ/download/box_logos.png)
