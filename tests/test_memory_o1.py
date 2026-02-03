import numpy as np

from bioplausible.kernel import EqPropKernel


def test_eqprop_kernel_memory_o1():
    """
    Verify that EqPropKernel does not store the full trajectory by default,
    confirming O(1) memory usage with respect to time steps.
    """
    input_dim = 10
    hidden_dim = 20
    output_dim = 5
    max_steps = 15
    batch_size = 4

    kernel = EqPropKernel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        max_steps=max_steps,
        use_gpu=False,  # Test on CPU
    )

    x = np.random.randn(batch_size, input_dim).astype(np.float32)

    # 1. Default (O(1) Memory Mode)
    # store_trajectory defaults to False
    h_star, act_log, info = kernel.solve_equilibrium(x)

    assert len(act_log) == 1, "O(1) memory mode should only return the final state"
    assert info["steps"] <= max_steps

    # 2. Trajectory Mode (O(T) Memory Mode)
    h_star_traj, act_log_traj, info_traj = kernel.solve_equilibrium(
        x, store_trajectory=True
    )

    # It might converge early, so check that it's consistent with steps taken
    steps_taken = info_traj["steps"]
    assert (
        len(act_log_traj) == steps_taken
    ), f"Trajectory mode should return {steps_taken} states, got {len(act_log_traj)}"

    # 3. Verify train_step calls
    # Mock compute_hebbian_update to check what it receives
    original_update = kernel.compute_hebbian_update
    received_logs = []

    def mock_update(act_free, act_nudged, x_input=None):
        received_logs.append((act_free, act_nudged))
        # Handle signature mismatch if original takes x_input or not
        try:
            return original_update(act_free, act_nudged, x_input)
        except TypeError:
            return original_update(act_free, act_nudged)

    kernel.compute_hebbian_update = mock_update

    y = np.random.randint(0, output_dim, size=(batch_size,))
    kernel.train_step(x, y)

    assert len(received_logs) == 1
    # Check that it passed single dicts (states), not lists
    act_free, act_nudged = received_logs[0]
    assert isinstance(act_free, dict)
    assert isinstance(act_nudged, dict)
    assert "h" in act_free

    print("O(1) Memory verification passed!")


if __name__ == "__main__":
    test_eqprop_kernel_memory_o1()
