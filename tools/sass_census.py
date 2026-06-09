#!/usr/bin/env python3
"""SASS op-mix census: mnemonic histogram per kernel for any cubin/binary.

Static counts by default; --weights <csv> (kernel_regex,addr_lo,addr_hi,trips
per line) scales regions by dynamic trip counts when the caller knows them.
Feeds tools/project.py (op-mix x table -> issue-cycle estimate).

Usage:
  sass_census.py BINARY [--kernel REGEX] [-o out.csv]
"""

import argparse
import collections
import re
import subprocess
import sys

CUOBJDUMP = "/opt/cuda-13.3/bin/cuobjdump"
INSTR_RE = re.compile(r"/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9.]*)")
FUNC_RE = re.compile(r"^\s*Function : (\S+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("--kernel", default=".*", help="regex on the mangled kernel name")
    ap.add_argument("--full", action="store_true",
                    help="keep full mnemonics (FFMA.FTZ) instead of bases (FFMA)")
    ap.add_argument("-o", "--out", default="-", help="output CSV path ('-' = stdout)")
    args = ap.parse_args()

    sass = subprocess.run([CUOBJDUMP, "-sass", args.binary],
                          capture_output=True, text=True, check=True).stdout
    kre = re.compile(args.kernel)

    counts = collections.defaultdict(collections.Counter)
    name = None
    for line in sass.splitlines():
        m = FUNC_RE.match(line)
        if m:
            name = m.group(1) if kre.search(m.group(1)) else None
            continue
        if name is None:
            continue
        m = INSTR_RE.search(line)
        if m:
            mn = m.group(1) if args.full else m.group(1).split(".")[0]
            counts[name][mn] += 1

    if not counts:
        print(f"no kernels matched {args.kernel!r}", file=sys.stderr)
        return 1

    out = sys.stdout if args.out == "-" else open(args.out, "w")
    out.write("kernel,op,count,share_pct\n")
    for kernel in sorted(counts):
        total = sum(counts[kernel].values())
        for op, n in counts[kernel].most_common():
            out.write(f"{kernel},{op},{n},{100.0 * n / total:.2f}\n")
    if out is not sys.stdout:
        out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
