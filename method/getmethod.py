from .core import Vanilla
from .mgda import MGDA
from .ew import EW
from .rlw import RLW
from .pcgrad import PCGrad
from .uw import UW

method_dict = {
    "vanilla" : Vanilla,
    'mgda' : MGDA,
    'ew' : EW,
    'rlw' : RLW,
    'pcgrad' : PCGrad,
    'uw' : UW
}