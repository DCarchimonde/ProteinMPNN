"""
N-methylation extension for ProteinMPNN
Support for designing proteins with N-methylated amino acids
"""

from .nmethyl_data_preprocessor import NmethylDataProcessor
from .train_nmethyl_mpnn import main as train_nmethyl
from .nmethyl_designer import NmethylDesigner

__all__ = [
    'NmethylDataProcessor',
    'train_nmethyl', 
    'NmethylDesigner'
]

__version__ = "1.0.0"