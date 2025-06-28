"""This is the evaluation script for the BIODCASE competition.
To run it on evaluation data, place the "Evaluation Set.zip" file
in ./data/01_raw before running the script.

The script reports:
- the average precision of the model on the evaluation data:
  "Average precision: 0.xxxxxx"
- the size of the model:
  "Model size:       XXXX bytes"
- The total time spent in preprocessing / feature extraction:
  "Total * n_windows = XXXXX"
- The total time spent in model inference:
  "total number of microseconds, XXXX"  (Note: the second time this line appears, the first one relates to preprocessing)
- the RAM usage of the model on the embedded target:
  "[RecordingMicroAllocator] Arena allocation total XXXX bytes"
"""
import json
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import click
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.metrics import AUC, Accuracy
from tensorflow.python.keras.utils.np_utils import to_categorical

from config import load_config
from data_preprocessing import run_preprocessing
from embedded_code_generation import create_target, generate_and_flash
from feature_extraction import run_feature_extraction
from paths import CLIPS_DIR, EVAL_CLIPS_DIR, EVAL_ZIP_PATH, GEN_CODE_DIR

logger = logging.getLogger("biodcase_tiny")
logger.setLevel(logging.INFO)


def prepare_eval_clips(eval_clips_dir: Path, eval_zip_path: Path):
    if eval_clips_dir.exists():
        # we already prepared the data
        return
    if not eval_zip_path.exists():
        raise FileNotFoundError(f"Can't find evaluation data at {eval_zip_path}")

    with TemporaryDirectory() as tmp_folder:
        tmp_folder = Path(tmp_folder)

        with zipfile.ZipFile(eval_zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_folder)

        labels_path = tmp_folder / "Evaluation Set (renamed)" / "binary_labels.csv"
        df = pd.read_csv(labels_path)

        inner_zip_path = tmp_folder / "Evaluation Set (renamed)" / "Evaluation Set.zip"

        with zipfile.ZipFile(inner_zip_path, 'r') as zip_ref:
            zip_ref.extractall(eval_clips_dir)

        # Recreate structure used in training dataset to be able to reuse the already written pipeline code
        negatives_folder = eval_clips_dir / "Negatives"
        yellowhammer_folder = eval_clips_dir / "Yellowhammer"

        negatives_folder.mkdir(exist_ok=True)
        yellowhammer_folder.mkdir(exist_ok=True)

        # Loop through labels and move files accordingly
        for _, row in df.iterrows():
            filename = row['Filename']
            label = row['Label']
            source_file = Path(eval_clips_dir) / filename
            if label == 0:  # Move to Negatives folder

                destination = negatives_folder / filename
                shutil.move(str(source_file), str(destination))
            elif label == 1:  # Move to Yellowhammer folder
                destination = yellowhammer_folder / filename
                shutil.move(str(source_file), str(destination))
            else:
                raise FileNotFoundError(f"File {filename} not found in evaluation_clips")
        print(f"Evaluation clips dataset preparation complete.")


def make_reference_ds(features_prq_path, features_shape_json_path, reference_dataset_path, channel_order=None):
    with features_shape_json_path.open("r") as f:
        features_shape = json.load(f)
    data = pd.read_parquet(features_prq_path)

    features = np.array(data["features"].to_list()).reshape((-1, *features_shape, 1))
    if channel_order is not None:
        features = np.transpose(features, channel_order)
    one_hot_labels = to_categorical(data["label"], num_classes=2)
    dataset = tf.data.Dataset.from_tensor_slices((features, one_hot_labels)).batch(32)
    dataset.save(str(reference_dataset_path))


def eval_model(
        features_shape_json_path: Path,
        features_prq_path: Path,
        tflite_model_path: Path,
        channel_order=None,
):
    data = pd.read_parquet(features_prq_path)
    data = data[data["split"].isin(["test", None])]  # data split None: eval clips
    with features_shape_json_path.open("r") as f:
        features_shape = json.load(f)
    input_data = np.array(data["features"].to_list()).reshape(-1, *features_shape, 1)
    if channel_order is not None:
        input_data = np.transpose(input_data, channel_order)

    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_dtype = input_details[0]["dtype"]
    output_dtype = output_details[0]["dtype"]

    if input_dtype == np.int8 and output_dtype == np.int8:
        input_scale = input_details[0]['quantization_parameters']['scales'][0]
        input_zero_point = input_details[0]['quantization_parameters']['zero_points'][0]
        output_scale = output_details[0]['quantization_parameters']['scales'][0]
        output_zero_point = output_details[0]['quantization_parameters']['zero_points'][0]
        input_data = np.clip(input_data / input_scale + input_zero_point, -128, 127).astype(input_dtype)
        quantized = True
    elif input_dtype == np.float32 and output_dtype == np.float32:
        quantized = False
    else:
        raise NotImplementedError(
            f"Case where input dtype is {input_dtype} "
            f"and output dtype is {output_dtype} not supported"
        )

    y_preds = []
    for sample in input_data:
        sample = sample.reshape(1, *sample.shape)  # tflite still needs batch dim
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        y_pred = interpreter.get_tensor(output_details[0]['index'])
        if quantized:
            y_pred = (y_pred.astype(np.float32) - output_zero_point) * output_scale
        y_preds.append(y_pred[0])
    y_preds = np.array(y_preds)
    y_true = to_categorical(data["label"].values, num_classes=2)
    average_precision = AUC(curve='PR')
    average_precision.update_state(y_true, y_preds)
    print(f"Average precision: {average_precision.result().numpy()}")


@click.command()
@click.option('--model-path', '-m',
              type=click.Path(exists=True, path_type=Path),
              required=True,
              help='Path to the model file (.keras or .tflite)')
@click.option('--config-path', '-c',
              type=click.Path(exists=True, path_type=Path),
              required=True,
              help='Path to the pipeline configuration YAML file')
@click.option('--dataset', '-d',
              type=click.Choice(['test', 'eval']),
              default='eval',
              help="Dataset to use for evaluation. Test dataset there to replicate participant's reported performance")
@click.option('--channel-order',
              type=str,
              default='0,1,2,3',
              help='Channel order as comma-separated values (e.g., "0,1,2,3")')
@click.option('--evaluate-embedded/--no-evaluate-embedded',
              default=True,
              help='Whether to evaluate on embedded target')
@click.option('--quantize',
              is_flag=True,
              help='Whether to quantize the model')
def run_evaluation(model_path, config_path, dataset, channel_order, evaluate_embedded, quantize):
    channel_order = tuple(int(x) for x in channel_order.split(','))
    config = load_config(config_path)

    if dataset == "eval":
        clips_dir = EVAL_CLIPS_DIR
        prepare_eval_clips(
            eval_clips_dir=clips_dir,
            eval_zip_path=EVAL_ZIP_PATH,
        )
    else:
        clips_dir = CLIPS_DIR

    with tempfile.TemporaryDirectory(prefix="biodcase") as tmpdir:
        root_dir = Path(tmpdir)
        preproc_prq_path = root_dir / "preproc.parquet"
        features_prq_path = root_dir / "features.parquet"
        features_shape_json_path = root_dir / "features_shape.json"
        features_sample_plot_path = root_dir / "features_sample_plot.png"
        reference_dataset_path = root_dir / "reference_dataset"

        run_preprocessing(
            config,
            clips_dir=clips_dir,
            preproc_prq_path=preproc_prq_path
        )
        run_feature_extraction(
            config,
            preproc_prq_path=preproc_prq_path,
            features_prq_path=features_prq_path,
            features_sample_plot_path=features_sample_plot_path,
            features_shape_json_path=features_shape_json_path,
        )
        if model_path.suffix == ".keras":
            make_reference_ds(
                features_prq_path=features_prq_path,
                features_shape_json_path=features_shape_json_path,
                reference_dataset_path=reference_dataset_path,
                channel_order=channel_order,
            )

        with tempfile.NamedTemporaryFile() as model_file:
            target = create_target(model_path, reference_dataset_path, config, quantize=quantize)
            model_file.write(target.get_model_buf())
            tflite_model_path = Path(model_file.name)
            tf.lite.experimental.Analyzer.analyze(model_path=str(tflite_model_path))
            eval_model(
                features_shape_json_path=features_shape_json_path,
                features_prq_path=features_prq_path,
                tflite_model_path=tflite_model_path,
                channel_order=channel_order,
            )
            if evaluate_embedded:
                generate_and_flash(config, target, GEN_CODE_DIR)


if __name__ == '__main__':
    run_evaluation()