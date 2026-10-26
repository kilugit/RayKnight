# ♟️ RayKnight Chess

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED.svg?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![PyQt6](https://img.shields.io/badge/PyQt-6-41CD52.svg?logo=qt&logoColor=white)](https://riverbankcomputing.com/software/pyqt/)

> **RayKnight Chess** is a deep learning chess engine and training suite built around custom ray and knight shift spatial operators (`ChessNet`).

---

## 💻 Local Setup & Installation

### 1. Prerequisites
- **Python:** 3.10 - 3.12
- **Git**
- **Stockfish Engine:** (Required for dataset generation, evaluation, and GUI engine matches)

---

### 2. Environment Setup & Dependency Installation

#### A. Create & Activate Virtual Environment
```bash
# Create virtual environment with Python 3.12:
py -3.12 -m venv venv

# Activate virtual environment:
# On Windows (PowerShell / Command Prompt):
.\venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate
```

#### B. Install Core Dependencies
Install the required base Python packages:
```bash
pip install --upgrade pip
pip install chess pyarrow pandas onnx tqdm PyQt6
```

#### C. Hardware Acceleration (PyTorch & ROCm / CUDA)
Install the optimized PyTorch and ONNX Runtime backend for your target hardware:

- **AMD GPU (Native ROCm on Windows):**  
  *Note: `device-gfx1200` corresponds to AMD Radeon RX 9060XT and related RDNA architectures.*
  ```bash
  python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "rocm[libraries,device-gfx1200]==10.0.0"
  python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1200]==2.13.0+rocm10.0.0" "torchvision[device-gfx1200]==0.28.0+rocm10.0.0" "torchaudio==2.11.0.2+rocm10.0.0"
  ```

- **ONNX Runtime (Windows ML):**  
  Provides DirectML (`DmlExecutionProvider`) and CPU execution providers:
  ```bash
  pip install onnxruntime-windowsml
  ```

- **NVIDIA GPU (CUDA):**
  ```bash
  # PyTorch with CUDA 12.1:
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

  # ONNX Runtime with CUDA / TensorRT:
  pip install onnxruntime-gpu
  ```

- **CPU-Only:**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install onnxruntime
  ```

#### D. Verify Installation
Run this quick check to confirm PyTorch device recognition and ONNX Runtime providers:
```bash
python -c "import torch, chess, onnxruntime as ort; print(f'PyTorch: {torch.__version__} | CUDA/ROCm Available: {torch.cuda.is_available()}'); print(f'ONNX Runtime: {ort.__version__} | Providers: {ort.get_available_providers()}'); print(f'python-chess: {chess.__version__}')"
```

---

### 3. Stockfish Engine Setup
RayKnight Chess uses Stockfish for self-play training data generation, position evaluation, and benchmark matches in the GUI. Because binaries are excluded from version control, set up the executable locally:

1. **Download Stockfish:**
   - Visit the official Stockfish repository at [https://github.com/official-stockfish/stockfish](https://github.com/official-stockfish/stockfish) and navigate to [Releases](https://github.com/official-stockfish/stockfish/releases).
   - Download the release archive matching your operating system and CPU (for 64-bit Windows, `stockfish-windows-x86-64-universal.zip` or the AVX2 build).

2. **Extract the Archive:**
   - Unpack the downloaded `.zip` (or `.tar`) archive.

3. **Place the Binary in the Project:**
   - Place the extracted executable into the `stockfish/` directory in the repository root:
     ```text
     ChessAI/
     ├── stockfish/
     │   └── stockfish-windows-x86-64-universal.exe   <-- Place executable here
     ├── chess_ai/
     ├── checkpoints/
     ├── dataset/
     └── ...
     ```
   > [!NOTE]
   > The default configured binary path in the codebase is `stockfish/stockfish-windows-x86-64-universal.exe`. You can select any custom Stockfish executable path directly in the GUI (using the **Browse...** button in the Dataset Generation or Model vs Stockfish Match tabs), set the `STOCKFISH_PATH` environment variable, or pass `--stockfish-path` to CLI scripts.

4. **Verify Stockfish Setup:**
   Run a quick check to verify Stockfish is detected and responding to UCI commands:
   ```bash
   python -c "from chess_ai.stockfish_runner import StockfishRunner; sf = StockfishRunner(); print('Stockfish successfully initialized!'); sf.close()"
   ```
   Or run the runner test suite:
   ```bash
   pytest tests/test_stockfish_runner.py
   ```

---

### 4. Launch Local Desktop GUI
The PyQt6 desktop application provides tabs for Play & Analysis (MCTS), Model vs Stockfish matches, live training with loss curves, Stockfish dataset generation, and checkpoint management.

```bash
# Quick launcher on Windows:
run.bat

# Or launch directly with Python:
python -m chess_ai.gui.app
```

---

### 5. Command Line Interface (CLI)

```bash
# Run UCI engine mode (compatible with Arena, Cutechess, BanksiaGUI)
python -m chess_ai.uci --model checkpoints/model.onnx

# Train a model on a Parquet dataset
python -m chess_ai.train --parquet-path dataset/stockfish_selfplay.parquet --preset fast --max-steps 5000

# Generate self-play dataset using Stockfish
python -m chess_ai.generate_dataset --games 50 --output dataset/stockfish_selfplay.parquet

# Export PyTorch checkpoint to ONNX format
python -m chess_ai.export_onnx --checkpoint checkpoints/best.pt --output checkpoints/model.onnx

# Run performance benchmarks and regression checks
python -m chess_ai.benchmark
```

---

## 🙏 Acknowledgments

- [Stockfish](https://github.com/official-stockfish/stockfish) — I gratefully thank the official Stockfish project and its contributors (https://github.com/official-stockfish/stockfish) for providing the open-source chess engine used for position evaluation, dataset self-play generation, and benchmark matches.
- [Chess games dataset by Angel Uriot](https://github.com/angeluriot/Chess_games) — I sincerely thank Angel Uriot (https://github.com/angeluriot/Chess_games) for creating and curating the high-level chess games dataset that supports machine learning training across this project.


