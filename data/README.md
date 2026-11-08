# Public data included in this repository

This folder contains **metadata and researcher-authored evaluation records**, not copies of the original University of East London policy documents.

- `corpus_manifest.csv` records document titles, source URLs, access dates and SHA-256 hashes.
- `questions_metadata.csv` contains the 120 researcher-authored questions, reference answers and split/category labels, but omits copied source passages.
- `evidence_metadata.csv` preserves evidence identifiers, document references and stable cleaned-text offsets while omitting the original evidence text.
- `question_distribution.csv` summarises the development/held-out category composition.

The original 15 source documents are deliberately not redistributed here. A full rerun requires reconstructing the frozen corpus from the manifest and verifying the recorded hashes.
