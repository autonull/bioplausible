"""
Cloud Resources Guide

Information on cloud providers for running Bio-Plausible experiments.
"""

CLOUD_PROVIDERS = [
    {
        "name": "Lambda Labs",
        "url": "https://lambdalabs.com/",
        "description": "Best price/performance for single GPUs. One-click Jupyter notebooks.",
        "tiers": [
            {"gpu": "1x A10", "price": "$0.60/hr", "vram": "24 GB"},
            {"gpu": "1x A100", "price": "$1.10/hr", "vram": "40 GB"},
            {"gpu": "1x H100", "price": "$2.49/hr", "vram": "80 GB"},
        ],
        "setup_cmd": "pip install bioplausible && python -m bioplausible_ui --headless",
    },
    {
        "name": "RunPod",
        "url": "https://runpod.io/",
        "description": "Community cloud with wide variety of GPUs. Supports Docker containers.",
        "tiers": [
            {"gpu": "1x RTX 3090", "price": "$0.29/hr", "vram": "24 GB"},
            {"gpu": "1x RTX 4090", "price": "$0.44/hr", "vram": "24 GB"},
            {"gpu": "1x A100", "price": "$1.69/hr", "vram": "80 GB"},
        ],
        "setup_cmd": "git clone https://github.com/bioplausible/bioplausible.git && cd bioplausible && pip install -r requirements.txt",
    },
    {
        "name": "Vast.ai",
        "url": "https://vast.ai/",
        "description": "P2P marketplace for GPU rentals. Usually the cheapest options.",
        "tiers": [
            {"gpu": "1x RTX 3060", "price": "$0.10/hr", "vram": "12 GB"},
            {"gpu": "1x RTX 3090", "price": "$0.20/hr", "vram": "24 GB"},
        ],
        "setup_cmd": "apt update && apt install -y python3-pip git && pip3 install torch torchvision",
    },
    {
        "name": "Google Colab",
        "url": "https://colab.research.google.com/",
        "description": "Free T4 GPUs. Pro version available for A100s.",
        "tiers": [
            {"gpu": "1x T4", "price": "Free", "vram": "16 GB"},
            {"gpu": "1x A100", "price": "$9.99/mo (Pro)", "vram": "40 GB"},
        ],
        "setup_cmd": "!pip install git+https://github.com/bioplausible/bioplausible.git",
    },
]

DEPLOYMENT_TIPS = """
**Quick Start Guide:**

1. **Choose a Provider:**
   - For quick testing: Google Colab (Free).
   - For heavy training: Lambda Labs or RunPod (Best Value).

2. **Environment Setup:**
   - Most cloud GPU instances come with PyTorch pre-installed.
   - You just need to install this package:
     `pip install bioplausible`

3. **Running Headless:**
   - If you are connecting via SSH, run the worker in 'headless' mode (without UI):
     `python -m bioplausible.p2p.worker --join <COORDINATOR_URL>`

4. **Persistence:**
   - Remember to save your results to a persistent volume or download them before terminating the instance!
   - The P2P network automatically uploads results, so local storage is less critical.
"""
