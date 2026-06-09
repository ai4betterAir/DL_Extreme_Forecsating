"""
Helpers to resolve and instantiate model classes dynamically.
"""

import inspect


PREFERRED_CLASS_NAMES = [
    "Sparse_LSTM_Class",
    "CNN_LSTM_BNN_Class",
    "CNN_LSTM_Class",
    "CNN_LSTM_generic_Class",
    "CNN_LSTM_BNN_generic_Class",
]


def resolve_model_class(module):
    """
    Return the most likely model class defined in a module.
    """
    explicit_class = getattr(module, "MODEL_CLASS", None)
    if inspect.isclass(explicit_class):
        return explicit_class
    if isinstance(explicit_class, str) and hasattr(module, explicit_class):
        candidate = getattr(module, explicit_class)
        if inspect.isclass(candidate):
            return candidate

    explicit_name = getattr(module, "MODEL_CLASS_NAME", None)
    if isinstance(explicit_name, str) and hasattr(module, explicit_name):
        candidate = getattr(module, explicit_name)
        if inspect.isclass(candidate):
            return candidate

    for class_name in PREFERRED_CLASS_NAMES:
        candidate = getattr(module, class_name, None)
        if inspect.isclass(candidate) and candidate.__module__ == module.__name__:
            return candidate

    candidates = []
    for _, candidate in inspect.getmembers(module, inspect.isclass):
        if candidate.__module__ != module.__name__:
            continue
        if callable(getattr(candidate, "run_all", None)):
            candidates.append(candidate)

    if not candidates:
        raise AttributeError(
            "No runnable model class found in module {module}".format(
                module=getattr(module, "__name__", module),
            )
        )

    if len(candidates) == 1:
        return candidates[0]

    for class_name in PREFERRED_CLASS_NAMES:
        for candidate in candidates:
            if candidate.__name__ == class_name:
                return candidate

    return candidates[0]


def instantiate_model(module, configuration):
    """
    Instantiate the resolved model class from a module.
    """
    model_class = resolve_model_class(module)
    try:
        return model_class(configuration)
    except TypeError:
        return model_class(Configuration=configuration)
