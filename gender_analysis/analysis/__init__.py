__all__ = [
    'dependency_parsing',
    'dunning',
    'gender_frequency',
    'instance_distance',
    'metadata_visualizations',
    'proximity',
    'base_analyzers',
]

from .dependency_parsing import *
from .dunning import *
from .gender_frequency import *
from .instance_distance import *
from .metadata_visualizations import *
from .proximity import GenderProximityAnalyzer
from .base_analyzers import CorpusAnalyzer
