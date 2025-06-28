#   Copyright 2024 BirdNET-Team
#   Copyright 2024 fold ecosystemics
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Build target for korvo2_bird_logger template project.
The class will check that the operations from the input keras model are supported by tflite-micro.
It then converts the keras model to tflite, and prepares all the information needed by the project template
to generate the final code.
"""
import os
import re
import shutil
from copy import copy, deepcopy
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import Model
from tensorflow.data import Dataset
from tensorflow.lite.tools import visualize as tflite_vis
from jinja2 import Environment, FileSystemLoader
from biodcase_tiny.feature_extraction.feature_extraction import FeatureConstants, convert_constants

TEMPLATE_EXTENSION = "jinja"
TEMPLATE_DIR = Path(__file__).parent / "firmware"

def tflite_to_byte_array(tflite_file: Path):
    with tflite_file.open("rb") as input_file:
        buffer = input_file.read()
    return buffer


def parse_op_str(op_str):
    """Converts a flatbuffer operator string to a format suitable for Micro
    Mutable Op Resolver. Example: CONV_2D --> AddConv2D.

    This fn is adapted from tensorflow lite micro tools scripts:
    (https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/tools/gen_micro_mutable_op_resolver/generate_micro_mutable_op_resolver_from_model.py)
    """
    # Edge case for AddDetectionPostprocess().
    # The custom code is TFLite_Detection_PostProcess.
    op_str = op_str.replace("TFLite", "")
    word_split = re.split("[_-]", op_str)
    formatted_op_str = ""
    for part in word_split:
        if len(part) > 1:
            if part[0].isalpha():
                formatted_op_str += part[0].upper() + part[1:].lower()
            else:
                formatted_op_str += part.upper()
        else:
            formatted_op_str += part.upper()
    # Edge cases
    formatted_op_str = formatted_op_str.replace("Lstm", "LSTM")
    formatted_op_str = formatted_op_str.replace("BatchMatmul", "BatchMatMul")
    return formatted_op_str


def get_model_ops_and_acts(model_buf):
    """Extracts a set of operators from a tflite model.

    This fn is adapted from tensorflow lite micro tools scripts:
    (https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/tools/gen_micro_mutable_op_resolver/generate_micro_mutable_op_resolver_from_model.py)
    """
    custom_op_found = False
    operators_and_activations = set()
    data = tflite_vis.CreateDictFromFlatbuffer(model_buf)
    for op_code in data["operator_codes"]:
        if op_code["custom_code"] is None:
            op_code["builtin_code"] = max(op_code["builtin_code"], op_code["deprecated_builtin_code"])
        else:
            custom_op_found = True
            operators_and_activations.add(tflite_vis.NameListToString(op_code["custom_code"]))
    for op_code in data["operator_codes"]:
        # Custom operator already added.
        if custom_op_found and tflite_vis.BuiltinCodeToName(op_code["builtin_code"]) == "CUSTOM":
            continue
        operators_and_activations.add(tflite_vis.BuiltinCodeToName(op_code["builtin_code"]))  # will be None if unknown
    return operators_and_activations


class ESPTarget:
    def __init__(
        self,
        model: Model | bytes,
        feature_config: FeatureConstants,
        reference_dataset: Dataset | None = None,
        quantize: bool = False
    ):
        if isinstance(model, Model):   # from Keras
            if reference_dataset is None:
                raise ValueError("Reference dataset must be provided when a keras model is passed.")
            self._model_buf = self._create_model_buf(model, reference_dataset, quantize)
        elif isinstance(model, bytes): # from tflite serialized model
            if quantize:
                raise ValueError("A tflite model was provided but `quantize` set to True.")
            if reference_dataset is not None:
                raise ValueError("A tflite model was provided, but `reference_dataset` was provided too.")
            self._model_buf = deepcopy(model)
        self._model_ops = get_model_ops_and_acts(self._model_buf)
        self._feature_config_buf = self.get_feature_config_buf(feature_config)

    def validate(self):
        """Validate Target inputs, including the compatibility of model."""
        self.check_model_compatible()

    @classmethod
    def setup_template_environment(cls, template_dir):
        cls.env = Environment(loader=FileSystemLoader(template_dir))

    def process_target_templates(self, outdir: Path) -> None:
        self.validate()

        # get available projects and their root template folders
        self.setup_template_environment(TEMPLATE_DIR)

        # extract context to be passed to jinja render
        context = self.extract_context()

        # Render and save each template
        if not outdir.exists():
            raise ValueError(f"{str(outdir)} does not exist, please create it.")

        # Process each file in the template directory
        for template_name in self.env.list_templates():
            template_path = Path(template_name)
            if template_path.suffix.lstrip(".") == TEMPLATE_EXTENSION:
                # Render and save the template file
                template = self.env.get_template(template_name)
                output_path = outdir / template_path.with_suffix("")
                with output_path.open("w") as f:
                    f.write(template.render(context))
            else:
                # Copy non-template files directly
                src_path = TEMPLATE_DIR / template_name
                dst_path = outdir / template_name
                os.makedirs(dst_path.parent, exist_ok=True)
                shutil.copyfile(src_path, dst_path)

    @staticmethod
    def _create_model_buf(model: Model, reference_dataset: Dataset, quantize=True):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT] if quantize else []
        if quantize:
            converter.inference_input_type = tf.dtypes.int8
            converter.inference_output_type = tf.dtypes.int8
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter._experimental_disable_per_channel_quantization_for_dense_layers = True

        def representative_dataset_gen():
            for example_spectrograms, example_spect_labels in reference_dataset.take(10):
                for X, _ in zip(example_spectrograms, example_spect_labels):
                    # Add a `batch` dimension, so that the spectrogram can be used
                    yield [X[tf.newaxis, ...]]

        converter.representative_dataset = representative_dataset_gen
        model_buf = converter.convert()
        return model_buf

    def get_model_buf(self):
        return self._model_buf

    @staticmethod
    def get_feature_config_buf(feature_config: FeatureConstants) -> bytearray:
        return convert_constants(feature_config)

    def check_model_compatible(self):
        if None in self._model_ops:
            raise ValueError(
                "Model contains op(s) that can't be converted to tflite micro. "
                f"Known ops: {self._model_ops.difference({None})}"
            )

    def save_tflite(self, outdir: Path) -> None:
        with outdir.open("wb") as f:
            f.write(self._model_buf)

    def extract_context(self) -> dict:
        model_hex_vals = [hex(b) for b in self._model_buf]
        feature_config_hex_vals = [hex(b) for b in self._feature_config_buf]
        return {
            "feature_config": {"hex_vals": feature_config_hex_vals},
            "model": {"hex_vals": model_hex_vals}
        }
