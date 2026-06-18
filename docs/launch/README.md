# Launch Kit

This folder keeps the public launch assets for SynthHub.

## Current Launch State

- PyPI: https://pypi.org/project/synthhub/
- GitHub: https://github.com/tauptlab/synthhub
- Colab: https://colab.research.google.com/github/tauptlab/synthhub/blob/main/examples/quickstart.ipynb
- Demo GIF: `docs/launch/assets/synthhub-demo.gif`

GitHub topics verified on 2026-06-18:

- `differential-privacy`
- `synthetic-data`
- `privacy`
- `pandas`
- `machine-learning`

## Launch Checklist

- [x] Package is on PyPI.
- [x] README opens with a DP-first headline and install command.
- [x] Colab quickstart installs from PyPI.
- [x] Demo GIF shows backend switching.
- [x] GitHub topics are set for discovery.
- [x] GitHub Release is published.
- [ ] PyPI API token used for the first release is revoked or rotated.
- [ ] PyPI Trusted Publishing is configured for future tokenless releases.
- [ ] Launch posts are published.

## Recommended Launch Order

1. Revoke or rotate the temporary PyPI token used for the first upload.
2. Publish the GitHub Release using `github-release-v0.1.0.post1.md`.
3. Post the Show HN / Reddit / LinkedIn / X copy from `social-posts.md`.
4. Watch GitHub Issues and PyPI install reports for the first 24 hours.
5. Collect feedback into focused issues instead of expanding the MVP ad hoc.

## Positioning

Short version:

> SynthHub is a scikit-learn-like API for differentially private synthetic data.

Longer version:

> DP synthetic-data libraries are fragmented. SynthHub wraps existing DP engines
> behind one pandas-first interface, so users can switch methods, compare
> utility, and inspect privacy-accounting reports without rewriting their
> workflow.
