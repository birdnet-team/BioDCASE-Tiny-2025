import json
from functools import partial
import faulthandler

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow as pa
from numpy.lib.stride_tricks import sliding_window_view
from plotly.subplots import make_subplots
from tqdm.dask import TqdmCallback

from biodcase_tiny.feature_extraction.feature_extraction import process_window, make_constants
from config import load_config, Config
from paths import PREPROC_PRQ_PATH, FEATURES_PRQ_PATH, FEATURES_SAMPLE_PLOT_PATH, FEATURES_SHAPE_JSON_PATH


def plot_features_sample(sample: pd.DataFrame, features_shape):
    """Plot a few samples of features
    """
    fmin = sample["features"].apply(lambda x: x.min()).min()
    fmax = sample["features"].apply(lambda x: x.max()).max()

    fig = make_subplots(
        rows=len(sample) + 1,
        cols=2,
        vertical_spacing=0.05,
        subplot_titles=[x.split("/")[-1] for x in list(sample["path"]) for _ in range(2)],
    )
    for i, (idx, row) in enumerate(sample.iterrows()):
        path, audio_data, sample_rate = row["path"], row["data"], row["sample_rate"]
        fig.add_trace(
            go.Heatmap(
                z=np.array(row["features"]).reshape(features_shape).T,
                zmin=fmin,
                zmax=fmax,
                name=path,
                showscale=False,
            ),
            row=i + 1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(audio_data.shape[0])),
                y=audio_data,
                mode="lines",
                name=path,
                showlegend=False,
            ),
            row=i + 1,
            col=2,
        )
    fig.update_layout(height=500 * len(sample), title_text="Features")
    return fig


def get_features_shape(clip_len_ms, clip_sample_rate, window_len_samples, window_stride_samples, mel_n_channels):
    clip_len_samples = int((clip_len_ms / 1000) * clip_sample_rate)
    return [(clip_len_samples - window_len_samples) // window_stride_samples + 1, mel_n_channels]


def apply_windowed(data, window_len, window_stride, fn):
    v = sliding_window_view(data, window_len)[::window_stride]
    outs = []
    for row in v:
        outs.append(fn(row))
    return np.array(outs)


def run_feature_extraction(
        config: Config,
        preproc_prq_path=PREPROC_PRQ_PATH,
        features_prq_path=FEATURES_PRQ_PATH,
        features_sample_plot_path=FEATURES_SAMPLE_PLOT_PATH,
        features_shape_json_path=FEATURES_SHAPE_JSON_PATH,
):
    faulthandler.enable()
    dask.config.set({"dataframe.convert-string": False})
    data = dd.read_parquet(preproc_prq_path)
    fe_config = config.feature_extraction
    dp_config = config.data_preprocessing
    fc = make_constants(
        fe_config.window_len,
        dp_config.sample_rate,
        fe_config.window_scaling_bits,
        fe_config.mel_n_channels,
        fe_config.mel_low_hz,
        fe_config.mel_high_hz,
        fe_config.mel_post_scaling_bits
    )

    # this partial stuff is just a way to set all config parameters, so we have a function that only takes data as input
    do_windows_fn = partial(
        apply_windowed,
        fn=partial(
            process_window,
            hanning=fc.hanning_window,
            mel_constants=fc.mel_constants,
            fft_twiddle=fc.fft_twiddle,
            window_scaling_bits=fc.window_scaling_bits,
            mel_post_scaling_bits=fc.mel_post_scaling_bits
        ),
        window_len=fe_config.window_len,
        window_stride=fe_config.window_stride,
    )

    features_example = do_windows_fn(data["data"].head(1)[0])
    features_shape = features_example.shape
    with TqdmCallback(desc="Extracting features from preprocessed data"):
        data = data.map_partitions(
            lambda df: df.assign(features=df["data"].apply(
                lambda clip: do_windows_fn(clip).flatten(),
            )),
            meta=pd.DataFrame(
                dict(
                    **{c: data._meta[c] for c in data._meta},
                    features=pd.Series([], dtype=object),
                )
            )
        )
        sample = data.head(10)
        features_sample_fig = plot_features_sample(sample, features_shape)
        data: dd.DataFrame = data.drop("data", axis=1)  # remove original audio
        data.to_parquet(
            features_prq_path,
            schema={
                'features': pa.list_(pa.float32())   # make sure array is serialized correctly
            },
            write_index=False,
        )
    features_sample_fig.write_image(features_sample_plot_path)
    with features_shape_json_path.open("w") as f:
        json.dump(features_shape, f)  # we save the feature shape as rows are flattened, so we can recover later


if __name__ == "__main__":
    config = load_config()
    run_feature_extraction(config)
