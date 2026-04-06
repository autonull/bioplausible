#!/usr/bin/env python3
"""
Initialize Cognition knowledge base for Circle Packing.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from asi_evolve.cognition.cognition import Cognition
from asi_evolve.utils.structures import CognitionItem


def init_cognition():
    exp_dir = Path(__file__).parent
    cog = Cognition(storage_dir=exp_dir / "cognition_data")
    cog.reset()

    knowledge = [
        # Geometry & constraints
        CognitionItem(
            content="Hexagonal close packing achieves density π/(2√3) ≈ 0.9069 in infinite plane. In a unit square, edge effects reduce this. Refine best patterns with variable radii and careful corner/boundary handling.",
            source="Circle Packing Theory",
            metadata={"topic": "hexagonal_packing", "importance": "critical"}
        ),
        CognitionItem(
            content="Edge effects: circles near boundaries have reduced space. Place larger circles at corners and edges. Corner circles can have radius up to √2/2 times distance from corner. Use small epsilon in overlap/bound checks.",
            source="Boundary Optimization",
            metadata={"topic": "edge_effects", "importance": "high"}
        ),
        CognitionItem(
            content="Variable radii: optimal solutions for n=26 use circles of different sizes. No uniform-radius assumption. Larger at center/corners, smaller fill gaps.",
            source="Variable Radii Strategy",
            metadata={"topic": "variable_radii", "importance": "high"}
        ),
        # Numerical optimization
        CognitionItem(
            content="Constrained optimization with scipy.optimize.minimize (SLSQP): Objective maximize sum(radii). Constraints: no overlaps (dist >= r_i + r_j + epsilon), inside bounds. Start from high-scoring nodes (score ≥ 2.2) as initial guess.",
            source="Numerical Optimization Guide",
            metadata={"topic": "scipy_optimize", "importance": "critical"}
        ),
        CognitionItem(
            content="Multi-start optimization: 3–5 different initial configurations from best nodes, optimize each, keep best. Avoids local optima. Tight tolerances and maxiter 500–1000 for SLSQP.",
            source="Multi-Start Strategy",
            metadata={"topic": "multi_start", "category": "optimization"}
        ),
        CognitionItem(
            content="scipy.optimize.differential_evolution for global refinement when local optimization plateaus. Can refine radii only (lower dimensional) then polish with SLSQP.",
            source="Global Escape",
            metadata={"topic": "differential_evolution", "category": "optimization"}
        ),
        CognitionItem(
            content="Targeted improvements: do not rewrite the whole program. Modify optimizer settings, constraints (explicit inequalities vs penalties), multi-start count, or add a refinement stage.",
            source="Incremental Refinement",
            metadata={"topic": "targeted_improvements", "importance": "high"}
        ),
        # n=26 & target
        CognitionItem(
            content="For n=26 in unit square: target sum of radii 2.635 (AlphaEvolve benchmark). Central hexagon + outer layer patterns work well. Variable radii and corner optimization are critical.",
            source="n=26 Knowledge",
            metadata={"topic": "n26", "importance": "critical"}
        ),
        CognitionItem(
            content="AlphaEvolve achieved 2.635 with constructor + constrained optimization. Key: good initialization from geometric pattern, then numerical refinement with strict constraints.",
            source="AlphaEvolve (Nature 2025)",
            metadata={"topic": "alphaevolve", "importance": "critical"}
        ),
        # Troubleshooting
        CognitionItem(
            content="Plateau at ~2.3–2.4: try (1) different initial pattern from high-scoring node, (2) increase maxiter to 500–1000, (3) explicit inequality constraints instead of penalties, (4) multi-start 3–5, (5) differential_evolution then SLSQP polish.",
            source="Breaking Plateaus",
            metadata={"topic": "plateau", "category": "troubleshooting"}
        ),
        CognitionItem(
            content="Overlap/bound checks: use epsilon (e.g. 1e-8) in constraints to avoid numerical tangency issues. Ensure no constraint violations before reporting sum of radii.",
            source="Numerical Stability",
            metadata={"topic": "stability", "category": "troubleshooting"}
        ),
        CognitionItem(
            content="Code structure: keep construction and optimization separate. Use high-scoring node code as base; make incremental changes (diff-style) rather than full rewrites.",
            source="Guidelines",
            metadata={"topic": "guidelines", "importance": "high"}
        ),
    ]

    ids = cog.add_batch(knowledge)
    print(f"Added {len(ids)} knowledge items to cognition (Phase 2 focus)")
    print(f"Total items: {len(cog)}")

    print("\n--- Testing retrieval ---")
    for query in [
        "How to use scipy optimization for circle packing?",
        "How to break through plateau at 2.3?",
        "Variable radii and constraints for n=26",
    ]:
        results = cog.retrieve(query, top_k=2)
        print(f"\nSearch: '{query}'")
        for item, score in results:
            print(f"  [{score:.3f}] {item.content[:80]}...")

    print("\nCognition initialization complete.")


if __name__ == "__main__":
    init_cognition()
