"""等九次 varied-start 训练完成，接着跑三个固定起点对照，再统一分析。"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = len(json.loads((args.study / "manifest.json").read_text(encoding="utf-8"))["jobs"])
    last = None
    while True:
        status = json.loads((args.study / "status.json").read_text(encoding="utf-8"))
        if status["failed"]:
            raise RuntimeError(status)
        current = (len(status["completed"]), tuple(status["active"]))
        if current != last:
            print("WAIT", current, flush=True)
            last = current
        if current[0] == expected and not status["active"]:
            break
        time.sleep(5)
    scripts = Path(__file__).resolve().parent
    subprocess.run([sys.executable, "-u", str(scripts / "fixed_start_control.py"),
        "--source", str(args.study / "source"), "--root", str(args.study / "fixed-controls")], check=True)
    subprocess.run([sys.executable, "-u", str(scripts / "reset_study.py"), "analyze",
        "--study", str(args.study), "--output", str(args.output)], check=True)
    print("FOLLOWUP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
