__all__ = [
    'dependency_parsing',
    'dunning',
    'frequency',
    'instance_distance',
    'metadata_visualizations',
    'proximity',
    'base_analyzers',
]

from .base_analyzers import CorpusAnalyzer
from .dependency_parsing import *
from .dunning import *
from .frequency import GenderFrequencyAnalyzer
from .instance_distance import GenderDistanceAnalyzer
from .metadata_visualizations import *
from .proximity import GenderProximityAnalyzer
