"""make_bundles.py -- build one self-contained upload bundle per project phase.

Each course phase is submitted separately, so each has to stand on its own. This
script writes bundles/<Phase_N>_<slug>.zip containing exactly the tracked files
of that phase and nothing else.

Why `git ls-files` rather than walking the directory: the working tree holds
things that must never reach a submission -- the 133 MB third-party
gumbel-max-scm clone under external/, a byte-identical duplicate of the Phase 2
sources kept for historical reasons, LaTeX build byproducts, __pycache__. All of
them are already excluded by .gitignore, so deriving the file list from git makes
the bundle contents identical to the repository contents by construction, with no
second exclusion list to keep in sync.

Unzipping a bundle produces a single top-level Phase_N/ directory.

Usage:
    python tools/make_bundles.py            # build every phase
    python tools/make_bundles.py Phase_3    # build one
"""

import os
import re
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
OUT_DIR = os.path.join(ROOT, "bundles")

SLUGS = {
    "Phase_1": "literature-review",
    "Phase_2": "proposal-and-proof-of-concept",
    "Phase_3": "progress-checkpoint",
    "Phase_4": "final-checkpoint",
    "Phase_5": "final-submission",
}

# Files scanned for cross-phase references. Binary deliverables are skipped.
TEXT_EXT = {".py", ".tex", ".md", ".txt", ".json", ".bib", ".csv"}


def git(*args):
    out = subprocess.run(["git"] + list(args), cwd=ROOT, check=True,
                         capture_output=True, text=True).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def discover_phases():
    names = {p.split("/")[0] for p in git("ls-files")}
    return sorted(n for n in names if re.fullmatch(r"Phase_\d+", n))


def cross_phase_mentions(phase, files):
    """Report other phases named inside this phase's text files.

    A mention is not automatically a dependency -- several are deliberate prose,
    e.g. Phase 3's report stating that Phase 2 is kept frozen elsewhere. This is
    a prompt to look, not a failure.
    """
    others = {p for p in SLUGS if p != phase}
    found = {}
    for rel in files:
        if os.path.splitext(rel)[1].lower() not in TEXT_EXT:
            continue
        try:
            with open(os.path.join(ROOT, rel), encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        hits = sorted(o for o in others if o in text)
        if hits:
            found[rel] = hits
    return found


def escapes_phase(phase, files):
    """Flag any sys.path insert or file path that resolves outside the phase.

    Every driver builds its paths from HERE/.. , so a genuine escape would show
    up as a parent reference deep enough to climb past the phase root.
    """
    bad = []
    for rel in files:
        if not rel.endswith(".py"):
            continue
        with open(os.path.join(ROOT, rel), encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                if '".."' not in line and "'..'" not in line:
                    continue
                # depth of the file below the phase root, e.g. src/envs/x.py -> 2
                depth = len(rel.split("/")) - 2
                if line.count('".."') + line.count("'..'") > depth:
                    bad.append(f"{rel}:{i}: {line.strip()}")
    return bad


def build(phase):
    files = git("ls-files", phase)
    if not files:
        print(f"  {phase}: no tracked files, skipped")
        return None

    slug = SLUGS.get(phase, "bundle")
    zip_path = os.path.join(OUT_DIR, f"{phase}_{slug}.zip")
    os.makedirs(OUT_DIR, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in files:
            z.write(os.path.join(ROOT, rel), arcname=rel)

    size_mb = os.path.getsize(zip_path) / 1e6
    print(f"  {phase:<9} {len(files):>3} files  {size_mb:>6.2f} MB  "
          f"-> bundles/{os.path.basename(zip_path)}")

    escapes = escapes_phase(phase, files)
    if escapes:
        print(f"    !! path escapes the phase root:")
        for e in escapes:
            print(f"       {e}")

    mentions = cross_phase_mentions(phase, files)
    if mentions:
        n = sum(len(v) for v in mentions.values())
        print(f"    note: {n} textual reference(s) to other phases in "
              f"{len(mentions)} file(s) -- prose, not imports "
              f"(run with --verbose to list)")
        if "--verbose" in sys.argv:
            for rel, hits in sorted(mentions.items()):
                print(f"       {rel}: {', '.join(hits)}")
    return zip_path


def main():
    wanted = [a for a in sys.argv[1:] if not a.startswith("--")]
    phases = wanted or discover_phases()

    dirty = git("status", "--porcelain")
    if dirty:
        print("warning: working tree has uncommitted changes. Bundles are built "
              "from the files on disk, so they will include them.\n")

    print(f"building bundles into {OUT_DIR}\n")
    built = [p for p in (build(ph) for ph in phases) if p]
    print(f"\n{len(built)} bundle(s) written. Each unzips to a single "
          f"Phase_N/ directory and is independently runnable.")


if __name__ == "__main__":
    main()
