"""Config loading (no heavy imports, so data preparation runs without PyTorch)."""
import os
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path=None):
    path = path or os.environ.get("NUTRI_CONFIG") or os.path.join(ROOT, "config.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # make every path absolute relative to the repository root (unless already absolute)
    for k, v in cfg["paths"].items():
        if not os.path.isabs(v):
            cfg["paths"][k] = os.path.join(ROOT, v)
    cfg["models"]["base"] = cfg["base_model"]  # the base row always uses the configured base model
    for k, v in cfg["models"].items():
        if v.startswith("checkpoints/"):
            cfg["models"][k] = os.path.join(cfg["paths"]["checkpoints_dir"], v[len("checkpoints/"):])
    return cfg
