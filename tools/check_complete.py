#!/usr/bin/env python3
"""Audit the COVERAGE manifest against the generated table.

Rules:
  - a TICKED box ([x]) whose row ids are absent from tu102_ops.csv is a
    FAILURE (the manifest claims something the table does not hold);
  - an UNTICKED box ([ ]) with rows PRESENT is a warning (tick it);
  - unticked-and-absent is fine (known-open work).
Row ids appear in backticks and may carry brace expansions
(`alu.ffma.{lat,tput}`); presence is prefix-match (variants count).
Exit 1 on failures (CI gate).
"""

import csv
import itertools
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def expand(rowid):
    """alu.ffma.{lat,tput} -> [alu.ffma.lat, alu.ffma.tput]"""
    m = re.search(r"\{([^}]*)\}", rowid)
    if not m:
        return [rowid]
    out = []
    for opt in m.group(1).split(","):
        out.extend(expand(rowid[:m.start()] + opt.strip() + rowid[m.end():]))
    return out


def main():
    table_ids = set()
    with open(os.path.join(ROOT, "table", "tu102_ops.csv")) as f:
        for row in csv.DictReader(f):
            table_ids.add(row["row_id"])

    def present(rid):
        return any(t == rid or t.startswith(rid + ".") or t.startswith(rid)
                   for t in table_ids)

    failures, warnings, checked = 0, 0, 0
    family = ""
    for line in open(os.path.join(ROOT, "table", "COVERAGE.md")):
        h = re.match(r"^##\s+(.*)", line)
        if h:
            family = h.group(1)
            continue
        m = re.match(r"^- \[([ x~])\]\s+(.*)", line)
        if not m:
            continue
        ticked = m.group(1) == "x"
        body = m.group(2)
        ids = []
        for q in re.findall(r"`([a-z][a-z0-9_.{},]*)`", body):
            if "." in q:  # row ids have dots; bare names are prose
                ids.extend(expand(q))
        if not ids or "descoped" in body.lower():
            continue  # prose-only or documented-absence boxes
        checked += 1
        missing = [i for i in ids if not present(i)]
        if ticked and missing:
            print(f"FAIL [{family}] ticked but missing: {' '.join(missing)}")
            failures += 1
        elif not ticked and not missing:
            print(f"WARN [{family}] all rows present but box unticked: {ids[0]} ...")
            warnings += 1

    print(f"check_complete: {checked} boxes checked, {failures} failures, "
          f"{warnings} tick-me warnings")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
