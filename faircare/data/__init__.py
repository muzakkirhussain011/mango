# faircare/data/__init__.py
from .heart import load_heart
from .adult import load_adult
from .synth_health import make_synth
from .partition import dirichlet_partition
from .mimic_eicu import load_mimic_demo, load_eicu_subset
