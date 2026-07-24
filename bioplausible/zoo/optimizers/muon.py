"""Muon/Dion optimizers.

The real Muon/Dion update strategies live in `bioplausible.zoo.mep.optimizers.
strategies.update` and are registered by `bioplausible.zoo.mep._registration`
as part of the MEP preset registration. This module exists only to keep the
`zoo/optimizers` package importable as a unit; it intentionally registers
nothing itself to avoid duplicate `muon`/`dion` registry entries.
"""

__all__: list[str] = []
