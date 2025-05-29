from typing import Callable, Dict, List, Tuple, Union

from pandas import Series
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator
from mlops.utils.models.sklearn import load_class

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

def separar_modelos_metadata(
    trained_models: List[BaseEstimator],
    all_model_metadata: List[Dict[str, Union[str, BaseEstimator]]],
) -> Dict[str, Union[str, Dict[str, Union[str, BaseEstimator]]]]:
    resultado = {}

    for i, (model, metadata) in enumerate(zip(trained_models, all_model_metadata), start=1):
        # return class name like string
        resultado[f"modelo{i}"] = type(model).__module__ + "." + type(model).__name__
        resultado[f"metadata_modelo{i}"] = metadata

    return resultado


@data_exporter
def train(
    settings: Tuple[
        List[Dict[str, Union[bool, float, int, str]]],
        csr_matrix,
        Series,
        List[Dict[str, Union[Callable[..., BaseEstimator], str]]],
    ],
    **kwargs,
) -> Tuple[List[BaseEstimator], List[Dict[str, Union[str, Callable[..., BaseEstimator]]]]]:
    all_hyperparameters, X, y, all_model_metadata = settings

    trained_models = []

    for hyperparameters, model_info in zip(all_hyperparameters, all_model_metadata):
        cls_or_str = model_info['cls']
        if isinstance(cls_or_str, str):
            model_class = load_class(cls_or_str)
        else:
            model_class = cls_or_str

        model = model_class(**hyperparameters)
        model.fit(X, y)
        trained_models.append(model)

    resultado = separar_modelos_metadata(trained_models, all_model_metadata)
    return resultado

