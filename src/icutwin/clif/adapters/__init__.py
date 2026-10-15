"""Our own raw -> CLIF adapters, for datasets without a published CLIF ETL."""

from icutwin.clif.adapters.eicu import EicuAdapter
from icutwin.clif.adapters.hirid import HiridAdapter
from icutwin.clif.adapters.inspire import InspireAdapter
from icutwin.clif.adapters.nwicu import NwicuAdapter
from icutwin.clif.adapters.ricd import RicdAdapter
from icutwin.clif.adapters.sicdb import SicdbAdapter
from icutwin.clif.adapters.zigong import ZigongAdapter

ADAPTERS = {"eicu": EicuAdapter, "hirid": HiridAdapter, "inspire": InspireAdapter, "nwicu": NwicuAdapter, "sicdb": SicdbAdapter,
            "ricd": RicdAdapter, "zigong": ZigongAdapter}
