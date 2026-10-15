"""icutwin on the AutoDL host: round 3's teacher-forced calibration (``icutwin.eval.calibration``) on another split's
first windows: the training stays (members) or the tuning stays, against the held-out ones. Overfitting shows as
calibration on members and over-prediction of rare events on non-members.
    python calib_split.py <train|tuning|held_out> <calibration args...>"""
import sys

from icutwin.eval import calibration as cal
from icutwin.eval import privacy as pv

split = sys.argv[1]
_first = pv.first_windows
pv.first_windows = lambda root, ds, _split: _first(root, ds, split)
cal.main(sys.argv[2:])
