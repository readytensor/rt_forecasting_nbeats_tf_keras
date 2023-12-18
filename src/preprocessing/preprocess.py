import os
import random
import sys
from typing import Any, Dict, Tuple, Union, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


from preprocessing.pipeline import (
    create_preprocess_pipelines,
    train_pipeline,
    transform_inputs,
    save_pipeline,
    load_pipeline
)


TRAINING_PIPELINE_FILE_NAME = "training_pipeline.joblib"
INFERENCE_PIPELINE_FILE_NAME = "inference_pipeline.joblib"

def get_preprocessing_pipelines(
    data_schema: Any,
    data: pd.DataFrame,
    preprocessing_config: Dict,
    default_hyperparameters: Dict,
) -> Tuple[Pipeline, Pipeline, int]:
    """
    Get training and inference preprocessing pipeline. 

    Args:
        data_schema (Any): A dictionary containing the data schema.
        data (pd.DataFame): A pandas DataFrame containing the train data split.
        preprocessing_config (Dict): A dictionary containing the preprocessing params.
        default_hyperparameters (Dict): A dictionary containing hyperparameters.

    Returns:
        A tuple containing:
            training_pipeline (Pipeline)
            inference_pipeline (Pipeline)
            encoding length (int)
    """
    # get encode len
    encode_to_decode_ratio = default_hyperparameters["encode_to_decode_ratio"]
    encode_len = get_encode_len(data, data_schema, encode_to_decode_ratio)
    # whether to use exogenous variables
    use_exogenous = default_hyperparameters["use_exogenous"]
    # create training and inference pipelines
    training_pipeline, inference_pipeline = create_preprocess_pipelines(
        data_schema=data_schema,
        preprocessing_config=preprocessing_config,
        encode_len=encode_len,
        use_exogenous=use_exogenous,
    )
    return training_pipeline, inference_pipeline, encode_len


def fit_transform_with_pipeline(
        pipeline: Pipeline, data: pd.DataFrame) -> Tuple[Pipeline, np.ndarray]:
    """
    Fit the preprocessing pipeline and transform data.

    Args:
        pipeline (Pipeline): The preprocessing pipeline.
        data (pd.DataFrame): The data as a numpy array

    Returns:
        Pipeline: Fitted preprocessing pipeline.
        np.ndarray: transformed data as a numpy array.
    """
    trained_pipeline = train_pipeline(pipeline, data)
    transformed_data = transform_inputs(trained_pipeline, data)
    return trained_pipeline, transformed_data


def transform_data(
        preprocess_pipeline: Any,
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Union[pd.Series, None]]:
    """
    Transform the data using the preprocessing pipeline and target encoder.

    Args:
        preprocess_pipeline (Any): The preprocessing pipeline.
        target_encoder (Any): The target encoder.
        data (pd.DataFrame): The input data as a DataFrame (targets may be included).

    Returns:
        Tuple[pd.DataFrame, Union[pd.Series, None]]: A tuple containing the transformed
            data and transformed targets;
            transformed targets are None if the data does not contain targets.
    """
    transformed_inputs = transform_inputs(preprocess_pipeline, data)
    return transformed_inputs


def get_encode_len(train_data, data_schema, encode_to_decode_ratio):
    history_len = train_data[data_schema.time_col].nunique()
    decode_len = data_schema.forecast_length
    if history_len <= 2 * decode_len:
        raise ValueError(
            f"History length ({history_len}) must be at least 2x forecast"
            f" length ({decode_len})"
        )
    target_encode_len = int(decode_len * encode_to_decode_ratio)
    train_history_len = history_len - decode_len

    if train_history_len < target_encode_len:
        encode_len = train_history_len
    else:
        encode_len = target_encode_len
    return encode_len


def save_pipelines(
    training_pipeline: Any,
    inference_pipeline: Any,
    preprocessing_dir_path: str
) -> None:
    """
    Save the preprocessing pipeline and target encoder to files.

    Args:
        preprocess_pipeline: The preprocessing pipeline.
        target_encoder: The target encoder.
        preprocessing_dir_path (str): dir path where the pipeline and target encoder
            are saved
    """
    if not os.path.exists(preprocessing_dir_path):
        os.makedirs(preprocessing_dir_path)
    save_pipeline(
        pipeline=training_pipeline,
        file_path_and_name=os.path.join(preprocessing_dir_path, TRAINING_PIPELINE_FILE_NAME)
    )
    save_pipeline(
        pipeline=inference_pipeline,
        file_path_and_name=os.path.join(preprocessing_dir_path, INFERENCE_PIPELINE_FILE_NAME)
    )

def save_preprocessing_pipeline(
    preprocess_pipeline: Any, preprocessing_dir_path: str, pipeline_type: str
) -> None:
    """
    Save the preprocessing pipeline to files.

    Args:
        preprocess_pipeline: The preprocessing pipeline.
        preprocessing_dir_path (str): dir path where the pipeline is to be saved.
        pipeline_type (str): Type of pipeline to be saved. Can be either "training" or
                            "inference"

    """
    if not os.path.exists(preprocessing_dir_path):
        os.makedirs(preprocessing_dir_path)

    if pipeline_type == "training":
        file_name = TRAINING_PIPELINE_FILE_NAME
    else:
        file_name = INFERENCE_PIPELINE_FILE_NAME
    save_pipeline(
        pipeline=preprocess_pipeline,
        file_path_and_name=os.path.join(preprocessing_dir_path, file_name),
    )


def load_pipeline_of_type(
    preprocessing_dir_path: str,
    pipeline_type: str
) -> Pipeline:
    """
    Load the preprocessing pipeline and target encoder

    Args:
        preprocessing_dir_path (str): dir path where the pipeline and target encoder
            are saved
        pipeline_type (str): Type of pipeline to be loaded. Can be either "training" or
                            "inference"

    Returns:
        Pipeline: Loaded preprocessing pipeline.
    """
    if pipeline_type == "training":
        file_name = TRAINING_PIPELINE_FILE_NAME
    else:
        file_name = INFERENCE_PIPELINE_FILE_NAME
    preprocess_pipeline = load_pipeline(
        file_path_and_name=os.path.join(preprocessing_dir_path, file_name)
    )
    return preprocess_pipeline


def inverse_scale_predictions(predictions: np.ndarray, pipeline: Pipeline) -> np.ndarray:
    """
    Applies the inverse transformation of MinMax scaling to the predictions using the
    trained inference pipeline.

    Args:
        predictions (np.ndarray): The model's predictions, of shape [N, T, D].
        pipeline (Pipeline): The trained inference pipeline containing the MinMax scaler.

    Returns:
        np.ndarray: The inverse-scaled predictions.
    """
    # Retrieve the minmax_scaler from the pipeline
    if 'minmax_scaler' in pipeline.named_steps:
        minmax_scaler = pipeline.named_steps['minmax_scaler']
        return minmax_scaler.inverse_transform(predictions)
    raise ValueError("MinMaxScaler not found in the pipeline.")


def offset_future_covariates_per_series(history_data: pd.DataFrame,
                                        forecast_data: Optional[pd.DataFrame],
                                        forecast_length: int,
                                        data_schema: Any) -> pd.DataFrame:
    """
    Offsets the future covariates for each series in the historical data by the
    specified forecast length.
    If forecast data is provided, use it to fill the last forecast_length rows
    for each series.

    Args:
        history_data (pd.DataFrame): The historical data containing past observations
                                     for multiple series.
        forecast_data (Optional[pd.DataFrame]): The future covariates data for the 
                                                forecast period.
        forecast_length (int): The length of the forecast window.
        data_schema (DataSchema): An object that contains the schema of the data, 
                                  including lists of past and future covariate names.

    Returns:
        pd.DataFrame: The modified historical data with future covariates offset
                      for each series.
    """
    # Return data as-is if there are no future covariates
    if len(data_schema.future_covariates) == 0:
        return history_data

    # Offset future covariates for each series separately
    offset_history_data = history_data.copy()
    offset_history_data = offset_history_data.sort_values(
        by=[data_schema.id_col, data_schema.time_col]
    )
    for series_id, group in history_data.groupby(data_schema.id_col):
        for col in data_schema.future_covariates:
            offset_history_data.loc[group.index, col] = group[col].shift(-forecast_length)

    # If forecast data is provided, use it to fill the last forecast_length rows for each series
    if forecast_data is not None:
        forecast_data_copy = forecast_data.copy()
        forecast_data_copy = forecast_data_copy.sort_values(
            by=[data_schema.id_col, data_schema.time_col]
        )
        for series_id, group in forecast_data_copy.groupby(data_schema.id_col):
            for col in data_schema.future_covariates:
                last_index = \
                    offset_history_data[offset_history_data[data_schema.id_col] == series_id]\
                    .index[-forecast_length:]
                offset_history_data.loc[last_index, col] = group[col].values[:forecast_length]

    # fill nans with zero - these zeros dont matter, but null values cause issues so filling with zero
    for col in data_schema.future_covariates:
        offset_history_data[col] = offset_history_data[col].fillna(0)
    return offset_history_data
