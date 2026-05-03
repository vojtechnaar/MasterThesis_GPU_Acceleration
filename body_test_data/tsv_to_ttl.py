#!/usr/bin/env python3
"""Convert a 3-column TSV edge list into Turtle.

Expected input format:

    subject<TAB>predicate<TAB>object

The defaults are tailored to gpu-test.tsv:
- ttd.drug_D0I9AA becomes ttd_drug:D0I9AA
- UniProt-like accessions such as P19526 become uniprot:P19526
- interacts_with becomes rel:interacts_with
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import TextIO
from urllib.parse import quote


TTD_DRUG_PREFIX = "ttd.drug_"
PN_LOCAL_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
UNIPROT_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-[0-9]+)?$"
)


def prefixed_or_iri(prefix: str, base_uri: str, local: str) -> str:
    """Return prefix:local when safe, otherwise an encoded full IRI."""
    if PN_LOCAL_RE.fullmatch(local) and not local.endswith("."):
        return f"{prefix}:{local}"
    return f"<{base_uri}{quote(local, safe='')}>"


def entity_to_turtle(token: str, fallback_base_uri: str, ttd_drug_base_uri: str, uniprot_base_uri: str) -> str:
    token = token.strip()

    if token.startswith(TTD_DRUG_PREFIX):
        drug_id = token[len(TTD_DRUG_PREFIX) :]
        return prefixed_or_iri("ttd_drug", ttd_drug_base_uri, drug_id)

    if UNIPROT_RE.fullmatch(token):
        return prefixed_or_iri("uniprot", uniprot_base_uri, token)

    return prefixed_or_iri("res", fallback_base_uri, token)


def predicate_to_turtle(token: str, predicate_base_uri: str) -> str:
    token = token.strip()
    return prefixed_or_iri("rel", predicate_base_uri, token)


def write_prefixes(
    output_file: TextIO,
    fallback_base_uri: str,
    predicate_base_uri: str,
    ttd_drug_base_uri: str,
    uniprot_base_uri: str,
) -> None:
    output_file.write(f"@prefix ttd_drug: <{ttd_drug_base_uri}> .\n")
    output_file.write(f"@prefix uniprot: <{uniprot_base_uri}> .\n")
    output_file.write(f"@prefix rel: <{predicate_base_uri}> .\n")
    output_file.write(f"@prefix res: <{fallback_base_uri}> .\n\n")


def convert_tsv_to_ttl(
    input_path: Path,
    output_path: Path,
    fallback_base_uri: str,
    predicate_base_uri: str,
    ttd_drug_base_uri: str,
    uniprot_base_uri: str,
) -> int:
    written = 0

    with input_path.open("r", encoding="utf-8", newline="") as input_file, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as output_file:
        reader = csv.reader(input_file, delimiter="\t")
        write_prefixes(output_file, fallback_base_uri, predicate_base_uri, ttd_drug_base_uri, uniprot_base_uri)

        for line_number, row in enumerate(reader, start=1):
            if not row or all(not value.strip() for value in row):
                continue

            if len(row) != 3:
                raise ValueError(f"{input_path}:{line_number}: expected 3 columns, got {len(row)}")

            subject, predicate, object_ = row
            output_file.write(
                f"{entity_to_turtle(subject, fallback_base_uri, ttd_drug_base_uri, uniprot_base_uri)} "
                f"{predicate_to_turtle(predicate, predicate_base_uri)} "
                f"{entity_to_turtle(object_, fallback_base_uri, ttd_drug_base_uri, uniprot_base_uri)} .\n"
            )
            written += 1

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a 3-column TSV triple file into Turtle.")
    parser.add_argument("input", type=Path, help="Input TSV path, e.g. gpu-test.tsv")
    parser.add_argument("output", type=Path, help="Output Turtle path, e.g. gpu-test.ttl")
    parser.add_argument("--resource-base", default="https://example.org/resource/")
    parser.add_argument("--predicate-base", default="https://example.org/relation/")
    parser.add_argument("--ttd-drug-base", default="https://identifiers.org/ttd.drug/")
    parser.add_argument("--uniprot-base", default="http://purl.uniprot.org/uniprot/")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = convert_tsv_to_ttl(
        args.input,
        args.output,
        args.resource_base,
        args.predicate_base,
        args.ttd_drug_base,
        args.uniprot_base,
    )
    print(f"Wrote {written} triples to {args.output}")


if __name__ == "__main__":
    main()
