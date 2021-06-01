__all__ = [
    'dependency_parsing',
    'dunning',
    'gender_frequency',
    'instance_distance',
    'metadata_visualizations',
    'proximity',
    'text_analyzer',
]

from .dependency_parsing import *
from .dunning import *
from .gender_frequency import *
from .instance_distance import *
from .metadata_visualizations import *
from .proximity import GenderProximityAnalyzer
from .text_analyzer import CorpusAnalyzer
