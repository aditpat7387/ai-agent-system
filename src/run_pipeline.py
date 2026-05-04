from pathlib import Path
import subprocess
import sys
import argparse
from datetime import datetime, timezone
import yaml

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
DEFAULT_CONFIG = ROOT / "configs" / "pipeline_runner.yaml"


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_command(step: dict) -> list[str]:
    if step.get("module"):
        return [PYTHON, "-m", step["module"]]
    script = step.get("script")
    if not script:
        raise ValueError(f"Step {step.get('name', '<unknown>')} must define module or script")
    script_path = Path(script)
    if not script_path.is_absolute():
        script_path = ROOT / script_path
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")
    return [PYTHON, str(script_path)]


def run_step(step: dict, dry_run: bool = False) -> int:
    name = step["name"]
    enabled = step.get("enabled", True)
    if not enabled:
        print(f"[{datetime.now(timezone.utc).isoformat()}] SKIP {name} disabled")
        return 0

    try:
        cmd = build_command(step)
    except Exception as e:
        print(f"[{datetime.now(timezone.utc).isoformat()}] FAIL {name} {e}")
        return 1

    print(f"[{datetime.now(timezone.utc).isoformat()}] START {name}")
    if dry_run:
        print(" ".join(cmd))
        return 0

    proc = subprocess.run(cmd, cwd=ROOT)
    print(f"[{datetime.now(timezone.utc).isoformat()}] END {name} rc={proc.returncode}")
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path

    cfg = load_config(cfg_path)
    steps = cfg.get("steps", [])
    if not steps:
        raise SystemExit(f"No steps defined in pipeline config: {cfg_path}")

    print(f"Using pipeline config: {cfg_path}")
    for step in steps:
        rc = run_step(step, dry_run=args.dry_run)
        if rc != 0:
            raise SystemExit(rc)
    print("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()