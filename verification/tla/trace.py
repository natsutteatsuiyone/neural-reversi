from __future__ import annotations

from pathlib import Path
import re
import sys

HIDDEN = {"pc", "stack", "nsp", "i", "thisSp", "sa", "ss", "cand", "obsEpoch"}


def parse(block: str) -> dict[str, str]:
    out = {}
    for m in re.finditer(r"^/\\ (\w+) = (.*?)(?=^/\\ |\Z)", block, re.S | re.M):
        out[m.group(1)] = " ".join(m.group(2).split())
    return out


def main() -> None:
    log = Path(sys.argv[1]).read_text(encoding="utf-8")
    start = log.find("The behavior up to this point is")
    if start < 0:
        sys.exit("no counterexample in this log")
    for line in log[:start].splitlines():
        if line.startswith("Error:"):
            print(line)
    body = re.split(r"^@!@!@ENDMSG|^\d+ states generated", log[start:], flags=re.M)[0]
    states = re.split(r"^State \d+: ", body, flags=re.M)[1:]
    prev = None
    for k, s in enumerate(states, 1):
        head, _, rest = s.partition("\n")
        cur = parse(rest)
        pcs = dict(re.findall(r'(\d+) :> "(\w+)"', cur.get("pc", "")))
        action = head.strip("<>").split(" ")[0]
        if prev is None:
            print(f"{k:3}. initial state")
        else:
            moved = [t for t in pcs if pcs[t] != prev[1].get(t)] or ["?"]
            changed = [
                f"{name}={val}" for name, val in cur.items()
                if name not in HIDDEN and prev[0].get(name) != val
            ]
            print(f"{k:3}. T{','.join(moved)} {action}  " + "; ".join(changed))
        prev = (cur, pcs)


if __name__ == "__main__":
    main()
