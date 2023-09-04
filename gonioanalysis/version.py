__version__ = "0.7.1"


# Need for easily checking the scipy version because of API changes
# For example, change in KDTree.query arguments: i_jobs -> workers
from scipy import __version__ as used_scipy_version
used_scipy_version = tuple([int(v) for v in used_scipy_version.split('.')])
