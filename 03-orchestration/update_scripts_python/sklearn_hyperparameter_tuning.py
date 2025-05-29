from typing import Callable, Dict, List, Tuple, Union

from pandas import Series
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator

from mlops.utils.models.sklearn import load_class, tune_hyperparameters

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

from collections.abc import Iterable

def flatten_once(seq):
    for item in seq:
        if isinstance(item, list):
            yield from item
        else:
            yield item
@transformer
def hyperparameter_tuning(
    training_set: Dict[str, Union[Series, csr_matrix]],
    model_class_name: Union[str, List[str]],
    *args,
    **kwargs,
) -> Tuple[
    List[Dict[str, Union[bool, float, int, str]]],
    csr_matrix,
    Series,
    List[Dict[str, Union[str, Callable[..., BaseEstimator]]]],
]:
    if isinstance(model_class_name, list):
        if len(model_class_name) == 1 and isinstance(model_class_name[0], list):
            model_class_names = model_class_name[0]
        else:
            model_class_names = model_class_name
    elif isinstance(model_class_name, str):
        model_class_names = model_class_name.split(',')
    else:
        raise ValueError(f"Formato no válido para model_class_name: {model_class_name}")

    X, X_train, X_val, y, y_train, y_val, _ = training_set['build']
    if isinstance(model_class_name, str):
        model_class_names = [name.strip() for name in model_class_name.split(',')]
    elif isinstance(model_class_name, list):
        model_class_names = list(flatten_once(model_class_name))
        model_class_names = [name.strip() for name in model_class_names if isinstance(name, str)]
    else:
        raise ValueError(f"Formato no válido para model_class_name: {model_class_name}")

    all_hyperparameters = []
    all_model_metadata = []

    for model_name in model_class_names:
        model_name = model_name.strip()
        model_class = load_class(model_name)

        hyperparameters = tune_hyperparameters(
            model_class,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            max_evaluations=kwargs.get('max_evaluations'),
            random_state=kwargs.get('random_state'),
        )

        all_hyperparameters.append(hyperparameters)
        all_model_metadata.append(dict(cls=model_class, name=model_name))

    return all_hyperparameters, X, y, all_model_metadata
