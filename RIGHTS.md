# Rights and data provenance

The repository's MIT license covers its original software and documentation. It does not automatically cover the source dataset, publisher articles, model outputs, pretrained artifacts, or other third-party material.

This project downloads the existing public [GPF Matched-News CSV](https://github.com/kishordgupta/gpf-semantic-response-audit) using a pinned source and checksum. It does not republish the full raw CSV in this repository. The original `full_prompt` column contains supplied publisher news text; that column is not an input to the attribution classifier. Public access to source material and a software license do not establish third-party redistribution permission.

The original dataset's rights and quality notices remain applicable. Consult [its documentation](https://github.com/kishordgupta/gpf-semantic-response-audit) and the relevant rights holders when determining a reuse that requires permission. Cite the dataset and preprint as described in [CITATION.md](CITATION.md).

Stored aggregate metrics, confusion counts, and feature summaries describe this finite experimental corpus. They are not evidence about authorship of an arbitrary person's writing, proof of academic misconduct, or reliable identification of an undisclosed serving checkpoint. Avoid republishing unnecessary full text in demonstrations or prediction logs.
