import json
import platform
import sys

import numpy
import pandas
import pytest
import scipy
import sklearn
import torch
import yaml

print(json.dumps({
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "sys_prefix": sys.prefix,
    "torch": {"version": torch.__version__, "file": torch.__file__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()},
    "numpy": {"version": numpy.__version__, "file": numpy.__file__},
    "scipy": {"version": scipy.__version__, "file": scipy.__file__},
    "sklearn": {"version": sklearn.__version__, "file": sklearn.__file__},
    "pandas": {"version": pandas.__version__, "file": pandas.__file__},
    "yaml": {"version": yaml.__version__, "file": yaml.__file__},
    "pytest": {"version": pytest.__version__, "file": pytest.__file__},
}, indent=2))
