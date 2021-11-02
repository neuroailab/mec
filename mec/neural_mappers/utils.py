import numpy as np


def map_from_str(map_type):
    if map_type.lower() == "pls":
        from mec.neural_mappers import PLSNeuralMap

        return PLSNeuralMap
    elif map_type.lower() == "corr":
        from mec.neural_mappers import CorrNeuralMap

        return CorrNeuralMap
    elif map_type.lower() == "sklinear":
        from mec.neural_mappers import SKLinearNeuralMap

        return SKLinearNeuralMap
    else:
        raise ValueError(f"{map_type.lower()} is not supported.")
