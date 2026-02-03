import torch

from bioplausible.models.neural_cube import NeuralCube


def test_neural_cube_smoke():
    print("Testing NeuralCube smoke test...")
    model = NeuralCube(cube_size=4, input_dim=16, output_dim=4, max_steps=5)
    x = torch.randn(2, 16)
    out = model(x)
    print("Output shape:", out.shape)
    assert out.shape == (2, 4)
    print("PASS")


if __name__ == "__main__":
    test_neural_cube_smoke()
