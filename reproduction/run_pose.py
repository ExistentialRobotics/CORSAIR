"""Run the ShapeNet pose evaluator without a GUI."""
import importlib.util
import os
from pathlib import Path
import sys

from joblib import Parallel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    spec = importlib.util.spec_from_file_location("evaluation-shapenet", ROOT / "evaluation-shapenet.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    workers = int(os.environ.get("POSE_WORKERS", "3"))
    module.Parallel = lambda **kwargs: Parallel(
        n_jobs=workers, backend="loky", verbose=10, pre_dispatch=workers, inner_max_num_threads=1,
    )
    module.App._init_gui = lambda self: None
    module.App.visualize = lambda self: None
    module.App()


if __name__ == "__main__":
    main()
