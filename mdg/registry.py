from functools import wraps


EVAL_METHOD_REGISTRY = {}
MODEL_WRAPPER_REGISTRY = {}
DATASET_LOADER_REGISTRY = {}

EVAL_METHOD = "eval_method"
MODEL_WRAPPER = "model_wrapper"
DATASET_LOADER = "dataset_loader"


def optional_params(func):
    """Allow a decorator to be called without parentheses if no kwargs are given.

    parameterize is a decorator, function is also a decorator.
    """

    @wraps(func)
    def wrapped(*args, **kwargs):
        """If a decorator is called with only the wrapping function just execute the real decorator.
           Otherwise return a lambda that has the args and kwargs partially applied and read to take a function as an argument.

        *args, **kwargs are the arguments that the decorator we are parameterizing is called with.

        the first argument of *args is the actual function that will be wrapped
        """
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return func(args[0])
        return lambda x: func(x, *args, **kwargs)

    return wrapped


@optional_params
def register(cls, _type: str, _name: str):
    if _type == EVAL_METHOD:
        EVAL_METHOD_REGISTRY[_name] = cls
    elif _type == MODEL_WRAPPER:
        MODEL_WRAPPER_REGISTRY[_name] = cls
    elif _type == DATASET_LOADER:
        DATASET_LOADER_REGISTRY[_name] = cls
    else:
        raise RuntimeError(f"No suitable registry found for type {_type}")
    return cls
