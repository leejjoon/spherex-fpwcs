"""Star-free WCS repair for SPHEREx Level-2 spectral images.

The six SPHEREx detectors are rigidly mounted behind one telescope, so one telescope
attitude (plus a velocity-aberration scale) fixes the WCS of all six.  This package fits
that attitude to the six header WCS solutions of an exposure and regenerates the WCS of
any detector from a fixed focal-plane template.
"""
from .template import Template, load_template
from .solve import Attitude, fit_attitude
from .sip import fit_tan_sip
from .api import Result, solve, apply, check

__version__ = "0.1.0"
__all__ = ["Template", "load_template", "Attitude", "fit_attitude", "fit_tan_sip",
           "Result", "solve", "apply", "check", "__version__"]
