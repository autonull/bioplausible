# Circle Packing Problem (n=26)

## Problem Statement

Pack 26 circles in a unit square (1x1) to maximize the sum of their radii.

## Objective

Maximize: Σ(r_i) for i = 1 to 26

Subject to:
- All circles must be inside the unit square: [0, 1] × [0, 1]
- No two circles can overlap
- Each circle i has center (x_i, y_i) and radius r_i > 0

## Target Performance

The AlphaEvolve paper (Nature 2025) achieved a sum of radii = **2.635** for n=26 circles.

## Implementation Requirements

Your code should provide a function `construct_packing()` that returns:
```python
def construct_packing():
    """
    Returns:
        centers: np.array of shape (26, 2) - (x, y) coordinates
        radii: np.array of shape (26,) - radius of each circle
        sum_radii: float - sum of all radii
    """
    ...
    return centers, radii, sum_radii
```

## Evaluation Criteria

1. **Primary**: Sum of radii (higher is better)
2. **Validity**: All circles must be:
   - Inside unit square
   - Non-overlapping (minimum gap: 1e-6)
