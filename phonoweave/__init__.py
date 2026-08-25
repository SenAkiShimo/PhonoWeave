__version__ = "0.0.1"

from .periodicity import normalized_periodicity as _normalized_periodicity
from . import lateral_relevance as _lateral_relevance
from . import nasal_relevance as _nasal_relevance
from . import rhotic_relevance as _rhotic_relevance

_lateral_relevance._periodicity = _normalized_periodicity
_nasal_relevance._periodicity = _normalized_periodicity
_rhotic_relevance._periodicity = _normalized_periodicity
