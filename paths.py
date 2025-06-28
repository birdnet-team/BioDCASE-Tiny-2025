from pathlib import Path

PIPELINE_CONFIG_FILE = Path(__file__).parent / "pipeline_config.yaml"

DATA_DIR = Path(__file__).parent / 'data'

RAW_DATA_DIR = DATA_DIR / "01_raw"
CLIPS_DIR = RAW_DATA_DIR / "clips"

PREPROC_DIR = DATA_DIR / "02_intermediate"
PREPROC_PRQ_PATH = PREPROC_DIR / "preproc.parquet"

FEATURES_DIR = DATA_DIR / "03_features"
FEATURES_PRQ_PATH = FEATURES_DIR / "features.parquet"
FEATURES_SHAPE_JSON_PATH = FEATURES_DIR / "features_shape.json"
FEATURES_SAMPLE_PLOT_PATH = FEATURES_DIR / "features_sample.png"

MODELS_DIR = DATA_DIR / '04_models'
KERAS_MODEL_PATH = MODELS_DIR / 'model.keras'
TFLITE_MODEL_PATH = MODELS_DIR / 'model.tflite'
REFERENCE_DATASET_PATH = MODELS_DIR / "reference_dataset"

REPORTING_DIR = DATA_DIR / '05_reporting'
TENSORBOARD_LOGS_PATH = REPORTING_DIR / "tensorboard"
CM_FIG_PATH = REPORTING_DIR / "cm.png"

GEN_CODE_DIR = DATA_DIR / '06_generated_code'

EVAL_ZIP_PATH = RAW_DATA_DIR / "Evaluation Set.zip"
EVAL_CLIPS_DIR = RAW_DATA_DIR / "eval_clips"
