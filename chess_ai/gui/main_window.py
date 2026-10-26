import torch
from PyQt6.QtWidgets import QMainWindow, QTabWidget
from chess_ai.gui.play_tab import PlayTab
from chess_ai.gui.match_tab import MatchTab
from chess_ai.gui.train_tab import TrainTab
from chess_ai.gui.dataset_tab import DatasetTab
from chess_ai.gui.model_tab import ModelTab
from chess_ai.gui.styles import DARK_THEME_QSS
from chess_ai.model import ChessNet, MODEL_PRESETS, find_latest_checkpoint, load_model
from chess_ai.stockfish_runner import find_stockfish_binary

class MainWindow(QMainWindow):

    def __init__(self, model_path: str | None=None):
        super().__init__()
        self.setWindowTitle('RayKnight Chess')
        self.resize(1180, 800)
        self.setStyleSheet(DARK_THEME_QSS)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.active_checkpoint = model_path or find_latest_checkpoint()
        if self.active_checkpoint:
            try:
                self.model, meta = load_model(self.active_checkpoint, device=device)
                print(f'[Autoload] Successfully loaded checkpoint: {self.active_checkpoint} ({self.model.count_parameters():,} params)')
            except Exception as e:
                print(f'[Autoload] Could not load checkpoint {self.active_checkpoint}: {e}')
                self.model = ChessNet(**MODEL_PRESETS['fast']).to(device)
                self.active_checkpoint = None
        else:
            self.model = ChessNet(**MODEL_PRESETS['fast']).to(device)
        self.model.eval()
        if device == 'cuda':
            self.model.half()
        self.stockfish_path = find_stockfish_binary()
        self.tabs = QTabWidget()
        self.play_tab = PlayTab(self.model)
        self.match_tab = MatchTab(self.model)
        self.train_tab = TrainTab(self.model)
        self.dataset_tab = DatasetTab()
        self.model_tab = ModelTab(self.model, active_checkpoint=self.active_checkpoint)
        self.model_tab.checkpoint_loaded.connect(self._on_checkpoint_reloaded)
        self.train_tab.model_updated.connect(self._on_model_reloaded)
        self.dataset_tab.dataset_generated.connect(self._on_dataset_generated)
        self.dataset_tab.stockfish_path_changed.connect(self._on_stockfish_path_changed)
        self.match_tab.stockfish_path_changed.connect(self._on_stockfish_path_changed)
        self.tabs.addTab(self.play_tab, 'Play & Analysis')
        self.tabs.addTab(self.match_tab, 'Model vs Stockfish Match')
        self.tabs.addTab(self.train_tab, 'Training')
        self.tabs.addTab(self.dataset_tab, 'Dataset Generation')
        self.tabs.addTab(self.model_tab, 'Model Management')
        self.setCentralWidget(self.tabs)

    def _on_dataset_generated(self, path: str):
        if self.dataset_tab.chk_gen_autoselect.isChecked():
            self.train_tab.on_dataset_generated(path)
        else:
            self.train_tab._refresh_dataset_list()

    def _on_stockfish_path_changed(self, path: str):
        self.stockfish_path = path
        self.dataset_tab.set_stockfish_path(path)
        self.match_tab.set_stockfish_path(path)
        self.play_tab.set_stockfish_path(path)

    def _on_checkpoint_reloaded(self, path: str):
        self.active_checkpoint = path
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.model, _ = load_model(path, device=device)
            self.model.eval()
            if device == 'cuda':
                self.model.half()
            self._propagate_model_to_tabs()
        except Exception as e:
            print(f'Error reloading checkpoint: {e}')

    def _on_model_reloaded(self, model, path: str):
        self.model = model
        self.active_checkpoint = path
        self._propagate_model_to_tabs()

    def _propagate_model_to_tabs(self):
        self.play_tab.model = self.model
        self.play_tab.mcts.model = self.model
        self.match_tab.model = self.model
        self.train_tab.model = self.model
        self.model_tab.model = self.model
        self.model_tab.active_checkpoint = self.active_checkpoint
        self.model_tab.lbl_ckpt.setText(f'Loaded Weights: {self.active_checkpoint}')
        if getattr(self.model, 'is_onnx', False):
            self.model_tab.lbl_arch.setText(f'Architecture: Ray-Knight ONNX ({self.model.count_parameters():,} params)')
            self.model_tab.lbl_device.setText(f"Provider: {getattr(self.model, 'active_provider', 'ONNXRuntime')}")
        else:
            self.model_tab.lbl_arch.setText(f'Architecture: Ray-Knight ({self.model.count_parameters():,} params)')
            dev_str = 'cuda (GPU)' if torch.cuda.is_available() else 'cpu'
            self.model_tab.lbl_device.setText(f'Device: {dev_str} | Precision: Native FP16')

    def closeEvent(self, event):
        for tab in [self.play_tab, self.match_tab, self.train_tab, self.dataset_tab, self.model_tab]:
            try:
                tab.close()
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        event.accept()
