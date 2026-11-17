# International Stroke Trial corrected dataset, version 2

Original data: Sandercock, Peter; Niewada, Maciej; Czlonkowska, Anna (2011).
International Stroke Trial database (version 2). University of Edinburgh,
Department of Clinical Neurosciences. DOI: https://doi.org/10.7488/ds/104.
Source: https://datashare.ed.ac.uk/items/91709c88-d913-45d8-8019-380027133d54.

Contains information from the International Stroke Trial database (version 2),
made available under the Open Data Commons Attribution License (ODC-By 1.0):
https://opendatacommons.org/licenses/by/1-0/.
The original licence text is in LICENSE_ODC_BY.txt. These data are not relicensed
under the repository's MIT code licence.

Download with `python Extension/poc/fetch_ist.py` from the repository root.
The downloader verifies the corrected CSV against the original repository MD5
and records URLs and SHA-256 in provenance.json. The source CSV uses a legacy
single-byte encoding; the experiment reads it with latin1 and selects only
documented numeric/categorical analysis fields.

Use corrected version 2, not the earlier supplementary CSV: the repository
documents corrected event-date anonymization errors. Raw patient-level CSVs
are ignored by Git and excluded from the submission companion; the companion
contains reproducible download instructions and aggregate experimental records.
No credentials, re-identification, private patient data or linkage is used.

The experiment uses randomized aspirin allocation RXASP (not aspirin received),
derived death indicator ID14 and ascertainment flag SET14D. DDEAD is not a
14-day death indicator and is not substituted for ID14. The treatment is an
assignment policy in a factorial trial, marginal over randomized heparin.
