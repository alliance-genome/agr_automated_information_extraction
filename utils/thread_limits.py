"""Pin native BLAS/OpenMP libraries to a single thread (SCRUM-5781).

Import this module **before** numpy / scikit-learn / LightGBM / XGBoost / pyarrow
so the thread-count environment variables are read when those libraries load.

Why: the classifier trainer reads the ABC embedding parquets with pyarrow, whose
OpenMP runtime otherwise corrupts LightGBM's once ``RandomizedSearchCV`` forks
``loky`` worker processes over the wide sparse BoW feature matrix — a worker dies
with SIGSEGV (reproduced on cervino). Running one thread per fit, with search-level
(loky) parallelism across candidates, avoids the crash at no real throughput cost.

``setdefault`` is used so an explicit environment value still wins.
"""

import os

for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")
