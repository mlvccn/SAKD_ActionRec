from ._base import Vanilla
from .KD import KD
from .CrossKD import CrossKD 
from .DKD import DKD



distiller_dict = {
    "NONE": Vanilla,
    "KD": KD,
    "CrossKD ": CrossKD,
    "DKD": DKD,
}
