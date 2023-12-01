

from .analysing import MAnalyser, MAverager
from .orientation_analysis import OAnalyser
from .optic_flow import FAnalyser
from .transmittance_analysis import TAnalyser, RelativeTAnalyser
from .startpos_analysis import StartposAnalyser


analyser_classes = {
        'orientation': OAnalyser,
        'motion': MAnalyser,
        'flow': FAnalyser,
        'transmittance': TAnalyser,
        'relative-transmittance': RelativeTAnalyser,
        'startposition': StartposAnalyser}
