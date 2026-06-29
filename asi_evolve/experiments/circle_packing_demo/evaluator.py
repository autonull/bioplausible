"""
Evaluator for circle packing example (n=26) with improved timeout handling
"""

import importlib.util
import numpy as np
import time
import os
import signal
import subprocess
import tempfile
import traceback
import sys
import pickle


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")


def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle

    Returns:
        True if valid, False otherwise
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    # Check if radii are nonnegative and not nan
    for i in range(n):
        if radii[i] < 0:
            print(f"Circle {i} has negative radius {radii[i]}")
            return False
        elif np.isnan(radii[i]):
            print(f"Circle {i} has nan radius")
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-6 or x + r > 1 + 1e-6 or y - r < -1e-6 or y + r > 1 + 1e-6:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-6:  # Allow for tiny numerical errors
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True


def run_with_timeout(program_path, timeout_seconds=20):
    """
    Run the program in a separate process with timeout
    using a simple subprocess approach

    Args:
        program_path: Path to the program file
        timeout_seconds: Maximum execution time in seconds

    Returns:
        centers, radii, sum_radii tuple from the program
    """
    # Create a temporary file to execute
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        # Write a script that executes the program and saves results
        script = f"""
import sys
import numpy as np
import os
import pickle
import traceback

# Add the directory to sys.path
sys.path.insert(0, os.path.dirname('{program_path}'))

# Debugging info
print(f"Running in subprocess, Python version: {{sys.version}}")
print(f"Program path: {program_path}")

try:
    # Read the code file (Evolve framework generates code without .py extension)
    with open('{program_path}', 'r') as f:
        code = f.read()

    # Create a module and execute the code in it
    import types
    program = types.ModuleType('program')
    exec(code, program.__dict__)

    # Evolve framework only generates construct_packing() function
    # Call it directly instead of run_packing()
    print("Calling construct_packing()...")
    centers, radii, sum_radii = program.construct_packing()
    print(f"construct_packing() returned successfully: sum_radii = {{sum_radii}}")

    # Save results to a file
    results = {{
        'centers': centers,
        'radii': radii,
        'sum_radii': sum_radii
    }}

    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {temp_file.name}.results")

except Exception as e:
    # If an error occurs, save the error instead
    print(f"Error in subprocess: {{str(e)}}")
    traceback.print_exc()
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{'error': str(e)}}, f)
    print(f"Error saved to {temp_file.name}.results")
"""
        temp_file.write(script.encode())
        temp_file_path = temp_file.name

    results_path = f"{temp_file_path}.results"

    try:
        # Run the script with timeout
        process = subprocess.Popen(
            [sys.executable, temp_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode

            # Always print output for debugging purposes
            print(f"Subprocess stdout: {stdout.decode()}")
            if stderr:
                print(f"Subprocess stderr: {stderr.decode()}")

            # Still raise an error for non-zero exit codes, but only after printing the output
            if exit_code != 0:
                raise RuntimeError(f"Process exited with code {exit_code}")

            # Load the results
            if os.path.exists(results_path):
                with open(results_path, "rb") as f:
                    results = pickle.load(f)

                # Check if an error was returned
                if "error" in results:
                    raise RuntimeError(f"Program execution failed: {results['error']}")

                return results["centers"], results["radii"], results["sum_radii"]
            else:
                raise RuntimeError("Results file not found")

        except subprocess.TimeoutExpired:
            # Kill the process if it times out
            process.kill()
            process.wait()
            raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")

    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(results_path):
            os.unlink(results_path)


def evaluate(program_path):
    """
    Evaluate the program by running it once and checking the sum of radii

    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics
    """
    # Target value from the paper
    TARGET_VALUE = 2.635  # AlphaEvolve result for n=26

    try:
        # For constructor-based approaches, a single evaluation is sufficient
        # since the result is deterministic
        start_time = time.time()

        # Use subprocess to run with timeout
        centers, radii, reported_sum = run_with_timeout(
            program_path, timeout_seconds=600  # Single timeout
        )

        end_time = time.time()
        eval_time = end_time - start_time

        # Ensure centers and radii are numpy arrays
        if not isinstance(centers, np.ndarray):
            centers = np.array(centers)
        if not isinstance(radii, np.ndarray):
            radii = np.array(radii)

        # Check for NaN values before validation
        if np.isnan(centers).any() or np.isnan(radii).any():
            print("NaN values detected in solution")
            return {
                "sum_radii": 0.0,
                "target_ratio": 0.0,
                "validity": 0.0,
                "eval_time": float(time.time() - start_time),
                "combined_score": 0.0,
            }

        # Validate solution
        valid = validate_packing(centers, radii)

        # Check shape and size
        shape_valid = centers.shape == (26, 2) and radii.shape == (26,)
        if not shape_valid:
            print(
                f"Invalid shapes: centers={centers.shape}, radii={radii.shape}, expected (26, 2) and (26,)"
            )
            valid = False

        # Calculate sum
        sum_radii = np.sum(radii) if valid else 0.0

        # Make sure reported_sum matches the calculated sum
        if abs(sum_radii - reported_sum) > 1e-6:
            print(f"Warning: Reported sum {reported_sum} doesn't match calculated sum {sum_radii}")

        # Target ratio (how close we are to the target)
        target_ratio = sum_radii / TARGET_VALUE if valid else 0.0

        # Validity score
        validity = 1.0 if valid else 0.0

        # Combined score - higher is better
        # Use exponential weighting: closer to target gets exponentially higher reward
        # This makes each small improvement near the target more valuable
        if validity > 0:
            # exp(target_ratio) - 1, normalized to [0, 1] range
            # When target_ratio = 0: score = 0
            # When target_ratio = 1: score = 1
            combined_score = sum_radii
        else:
            combined_score = 0.0

        print(
            f"Evaluation: valid={valid}, sum_radii={sum_radii:.6f}, target={TARGET_VALUE}, ratio={target_ratio:.6f}, time={eval_time:.2f}s"
        )

        return {
            "sum_radii": float(sum_radii),
            "target_ratio": float(target_ratio),
            "validity": float(validity),
            "eval_time": float(eval_time),
            "combined_score": float(combined_score),
        }

    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        traceback.print_exc()
        return {
            "sum_radii": 0.0,
            "target_ratio": 0.0,
            "validity": 0.0,
            "eval_time": 0.0,
            "combined_score": 0.0,
        }


# Stage-based evaluation for cascade evaluation
def evaluate_stage1(program_path):
    """
    First stage evaluation - quick validation check
    """
    try:
        # Use the simplified subprocess approach
        try:
            centers, radii, sum_radii = run_with_timeout(program_path, timeout_seconds=600)

            # Ensure centers and radii are numpy arrays
            if not isinstance(centers, np.ndarray):
                centers = np.array(centers)
            if not isinstance(radii, np.ndarray):
                radii = np.array(radii)

            # Validate solution (shapes and constraints)
            shape_valid = centers.shape == (26, 2) and radii.shape == (26,)
            if not shape_valid:
                print(f"Invalid shapes: centers={centers.shape}, radii={radii.shape}")
                return {"validity": 0.0, "error": "Invalid shapes"}

            valid = validate_packing(centers, radii)

            # Calculate sum
            actual_sum = np.sum(radii) if valid else 0.0

            # Target from paper
            target = 2.635

            # Exponential combined score for stage 1, normalized to [0, 1]
            # Use exponential weighting to reward getting closer to target
            if valid:
                target_ratio = actual_sum / target
                combined_score = (np.exp(target_ratio) - 1) / (np.e - 1)
            else:
                combined_score = 0.0

            # Return evaluation metrics
            return {
                "validity": 1.0 if valid else 0.0,
                "sum_radii": float(actual_sum),
                "target_ratio": float(actual_sum / target if valid else 0.0),
                "combined_score": float(combined_score),
            }

        except TimeoutError as e:
            print(f"Stage 1 evaluation timed out: {e}")
            return {"validity": 0.0, "combined_score": 0.0, "error": "Timeout"}
        except Exception as e:
            print(f"Stage 1 evaluation failed: {e}")
            print(traceback.format_exc())
            return {"validity": 0.0, "combined_score": 0.0, "error": str(e)}

    except Exception as e:
        print(f"Stage 1 evaluation failed completely: {e}")
        print(traceback.format_exc())
        return {"validity": 0.0, "combined_score": 0.0, "error": str(e)}


def evaluate_stage2(program_path):
    """
    Second stage evaluation - full evaluation
    """
    # Full evaluation as in the main evaluate function
    return evaluate(program_path)


if __name__ == "__main__":
    """
    Command-line interface for Evolve framework
    Usage: python evaluator.py <code_file> <output_json>
    """
    import json

    if len(sys.argv) != 3:
        print("Usage: python evaluator.py <code_file> <output_json>")
        sys.exit(1)

    code_file = sys.argv[1]
    output_file = sys.argv[2]

    result = evaluate(code_file)

    # Add framework compatibility fields
    result["success"] = result.get("validity", 0.0) > 0
    result["score"] = result.get("combined_score", 0.0)
    result["eval_score"] = result.get("combined_score", 0.0)
    result["complexity"] = len(open(code_file).read()) if os.path.exists(code_file) else 0

    # Write results to JSON file
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    # Print summary
    if result.get("success", False):
        print(f"[OK] Valid packing: sum_radii = {result['sum_radii']:.6f}")
        print(f"  Target ratio: {result['target_ratio']:.6f}")
        print(f"  Combined score: {result['combined_score']:.6f}")
    else:
        error_msg = result.get("error", "Invalid packing")
        print(f"[FAIL] Invalid packing: {error_msg}")
