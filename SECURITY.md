# Security Policy

SynthHub is privacy-sensitive software. Please report suspected privacy,
accounting, or data-leakage issues privately before opening a public issue.

## Reporting

Use GitHub private vulnerability reporting when it is enabled for this
repository. If it is not enabled, contact the maintainers privately through the
TaupT Lab GitHub organization before opening a public issue.

Include:

- affected backend or adapter
- minimal reproduction steps
- expected privacy/accounting behavior
- observed behavior

Do not include private datasets in reports. Use synthetic or public data unless
we explicitly arrange another channel.

## Scope

In scope:

- epsilon/delta not being passed correctly
- `epsilon_spent` under-reporting
- optional backend DP-disable flags being allowed
- sample output leaking raw rows through adapter bugs
- schema or preprocessing behavior that silently changes the DP contract

Out of scope:

- mathematical flaws in upstream engines that SynthHub does not modify
- generic dependency CVEs without a SynthHub-specific exploit path
