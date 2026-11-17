"""Download the corrected public IST v2 dataset from the original repository.

Original data: Sandercock, Niewada and Czlonkowska (2011), DOI10.7488/ds/104.
ODC Attribution licence. No credentials or patient re-identification are used.
"""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/ist"
FILES = {
    "IST_corrected.csv": "eca52b56-6d1b-47b3-814a-0b7089f8c870",
    "IST_variables.csv": "069dc665-f3c8-4c95-8ec0-bf6736e0c8ac",
    "LICENSE_ODC_BY.txt": "4ea1fbdb-9b64-4172-aa85-aa0f89dab6c4",
}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    provenance = dict(dataset="International Stroke Trial database (version 2)",
                      doi="10.7488/ds/104", licence="ODC-By 1.0", files={})
    for name, identifier in FILES.items():
        url = f"https://datashare.ed.ac.uk/server/api/core/bitstreams/{identifier}/content"
        target = OUTPUT / name
        if not target.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            target.write_bytes(data)
        data = target.read_bytes()
        if name == "IST_corrected.csv" and hashlib.md5(data).hexdigest() != "c233a72eef0aa30b23569b2017ed17dd":
            raise ValueError("Corrected IST v2 content differs from original repository checksum")
        provenance["files"][name] = dict(url=url, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        print(f"Verified {name}: {len(data)} bytes", flush=True)
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
