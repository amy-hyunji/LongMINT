import os
import subprocess
import sys
from pathlib import Path

MEM_ALPHA_DIR = Path(__file__).resolve().parent

if not MEM_ALPHA_DIR.is_dir():
    raise SystemExit(
        f"Expected mem_alpha scaffolding at {MEM_ALPHA_DIR}. "
        "Run the MINTEval_dev setup first."
    )


def _resolve_agent_config(argv):
    out = list(argv)
    for i, tok in enumerate(out):
        if tok == "--agent_config" and i + 1 < len(out):
            out[i + 1] = str(Path(out[i + 1]).expanduser().resolve())
            return out
        if tok.startswith("--agent_config="):
            key, _, val = tok.partition("=")
            out[i] = f"{key}={Path(val).expanduser().resolve()}"
            return out
    return out


def main():
    argv = _resolve_agent_config(sys.argv[1:])

    if not any(a == "--agent_config" or a.startswith("--agent_config=") for a in argv):
        default_cfg = MEM_ALPHA_DIR / "config" / "memalpha_unified.yaml"
        if not default_cfg.is_file():
            raise SystemExit(
                f"--agent_config not given and no default at {default_cfg}"
            )
        argv = ["--agent_config", str(default_cfg)] + argv

    cmd = [sys.executable, str(MEM_ALPHA_DIR / "main.py")] + argv
    env = os.environ.copy()
    extra_path = str(MEM_ALPHA_DIR)
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = extra_path + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = extra_path

    sys.exit(subprocess.call(cmd, cwd=str(MEM_ALPHA_DIR), env=env))


if __name__ == "__main__":
    main()
