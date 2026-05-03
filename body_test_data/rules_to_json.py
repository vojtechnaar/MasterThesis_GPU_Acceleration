#!/usr/bin/env python3
"""
Convert a text-format rule file (.txt) to a JSON rule file (.json).

Text format (one rule per line, optional | comment at end):
  ( ?a pred ?b ) ^ ( ?b pred2 ?c ) => ( ?a pred3 ?c ) | Support: 5, ...

JSON format (array of rule objects matching the RDFRules export format):
  [{"body": [{"subject":{"type":"variable","value":"?a"},
              "predicate":"pred",
              "object":{"type":"variable","value":"?b"}}, ...],
    "head": {...}}]

Usage:
  python rules_to_json.py input.txt output.json
  python rules_to_json.py input.txt          # writes input.json
"""

import json
import sys
import re
from pathlib import Path


def split_atoms(body_text: str) -> list[str]:
    """Split body on ^ at paren depth 0."""
    atoms = []
    current = []
    depth = 0
    for ch in body_text:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == '^' and depth == 0:
            t = ''.join(current).strip()
            if t:
                atoms.append(t)
            current = []
        else:
            current.append(ch)
    t = ''.join(current).strip()
    if t:
        atoms.append(t)
    return atoms


def parse_term(token: str) -> dict:
    token = token.strip()
    if token.startswith('?'):
        return {'type': 'variable', 'value': token}
    return {'type': 'constant', 'value': token}


def parse_atom(atom_str: str) -> dict:
    inner = atom_str.strip()
    if inner.startswith('('):
        inner = inner[1:]
    if inner.endswith(')'):
        inner = inner[:-1]
    inner = inner.strip()

    # Split into exactly 3 tokens: subject predicate object
    # Predicate may be <uri> (no spaces inside) or prefix:local or bare word
    # Split on whitespace but keep <...> together
    tokens = re.findall(r'<[^>]*>|\S+', inner)
    if len(tokens) != 3:
        raise ValueError(f'Expected 3 tokens in atom, got {len(tokens)}: {atom_str!r}')
    subj_tok, pred_tok, obj_tok = tokens
    return {
        'subject':   parse_term(subj_tok),
        'predicate': pred_tok,
        'object':    parse_term(obj_tok),
    }


def parse_rule_line(line: str) -> dict:
    # Strip inline comment
    bar = line.find('|')
    if bar != -1:
        line = line[:bar]
    line = line.strip()
    if not line:
        return None

    arrow = line.find('=>')
    if arrow == -1:
        raise ValueError(f'Rule missing =>: {line!r}')

    body_text = line[:arrow].strip()
    head_text = line[arrow + 2:].strip()

    body = [parse_atom(a) for a in split_atoms(body_text)]
    head = parse_atom(head_text)
    return {'body': body, 'head': head}


def convert_file(src: Path, dst: Path) -> int:
    rules = []
    with open(src) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rule = parse_rule_line(line)
                if rule:
                    rules.append(rule)
            except ValueError as e:
                print(f'  Warning line {lineno}: {e}', file=sys.stderr)

    with open(dst, 'w') as f:
        json.dump(rules, f, indent=2)
    return len(rules)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix('.json')

    print(f'Converting {src} → {dst}')
    n = convert_file(src, dst)
    print(f'Written {n} rules to {dst}')


if __name__ == '__main__':
    main()
