from .core import Vanilla
from .mgda import MGDA
from .ew import EW

method_dict = {
    "vanilla" : Vanilla,
    'mgda' : MGDA,
    'ew' : EW
}