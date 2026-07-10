# Security Policy

## Supported versions

LOFOP is pre-1.0; security fixes are applied to the latest `0.1.x` release and
the development branch.

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |
| < 0.1   | No        |

## Reporting a vulnerability

Please report security issues privately rather than opening a public issue.

- Preferred: open a private advisory via GitHub Security Advisories
  ("Report a vulnerability" on the repository's Security tab).
- Alternatively, email the maintainer at `durgamani.d.e.c.e.50@gmail.com` with
  a description, affected version, and reproduction steps.

We aim to acknowledge reports within a few days and to provide a remediation
timeline after triage. Please give us a reasonable window to release a fix
before any public disclosure.

## Scope notes

LOFOP loads model checkpoints with `weights_only=True` and builds components
from declarative configs through the registry. Only load checkpoints and
configs from sources you trust: a malicious config can instantiate arbitrary
registered components, and native ops are compiled from the bundled C++ source
on first use.
