import os
import time
import torch
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit, QComboBox, QFileDialog, QMessageBox, QTextEdit
from chess_ai.model import ChessNet, find_latest_checkpoint, load_model, save_checkpoint
from chess_ai.export_onnx import export_onnx

class OnnxExportWorker(QThread):
    finished = pyqtSignal(bool, str)
    log = pyqtSignal(str)

    def __init__(self, model: ChessNet, output_path: str, half_precision: bool):
        super().__init__()
        import copy
        self.model = copy.deepcopy(model).to('cpu')
        self.output_path = output_path
        self.half_precision = half_precision

    def run(self):
        try:
            self.log.emit(f'Exporting model to ONNX: {self.output_path} (half={self.half_precision})...')
            out = export_onnx(self.model, output_path=self.output_path, half_precision=self.half_precision)
            size_mb = os.path.getsize(out) / (1024 * 1024)
            self.log.emit(f'ONNX export verified successfully! File size: {size_mb:.2f} MB')
            self.finished.emit(True, out)
        except Exception as e:
            self.log.emit(f'Export error: {e}')
            self.finished.emit(False, str(e))

class ModelTab(QWidget):
    checkpoint_loaded = pyqtSignal(str)

    def __init__(self, model: ChessNet, active_checkpoint: str | None=None, parent=None):
        super().__init__(parent)
        self.model = model
        self.active_checkpoint = active_checkpoint
        self.export_worker: OnnxExportWorker | None = None
        self._init_ui()
        self._refresh_checkpoints_table()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        info_group = QGroupBox('Active Model Status')
        info_layout = QVBoxLayout(info_group)
        row1 = QHBoxLayout()
        if getattr(self.model, 'is_onnx', False):
            self.lbl_arch = QLabel(f'Architecture: Ray-Knight ONNX ({self.model.count_parameters():,} params)')
            self.lbl_device = QLabel(f"Provider: {getattr(self.model, 'active_provider', 'ONNXRuntime')}")
        else:
            self.lbl_arch = QLabel(f'Architecture: Ray-Knight Operator Network ({self.model.count_parameters():,} params)')
            device = 'cuda (GPU)' if torch.cuda.is_available() else 'cpu'
            self.lbl_device = QLabel(f'Device: {device} | Precision: Native FP16')
        self.lbl_arch.setStyleSheet('font-weight: bold; font-size: 13px; color: #63b3ed;')
        row1.addWidget(self.lbl_arch)
        self.lbl_device.setAlignment(Qt.AlignmentFlag.AlignRight)
        row1.addWidget(self.lbl_device)
        info_layout.addLayout(row1)
        self.lbl_ckpt = QLabel(f"Loaded Weights: {self.active_checkpoint or 'Initialized (Untrained)'}")
        self.lbl_ckpt.setStyleSheet('color: #e2e8f0; font-family: monospace;')
        info_layout.addWidget(self.lbl_ckpt)
        layout.addWidget(info_group)
        ckpt_group = QGroupBox('Saved Checkpoints (checkpoints/ directory)')
        ckpt_layout = QVBoxLayout(ckpt_group)
        self.table_ckpts = QTableWidget(0, 4)
        self.table_ckpts.setHorizontalHeaderLabels(['Filename', 'Format', 'Size (MB)', 'Last Modified'])
        self.table_ckpts.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table_ckpts.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table_ckpts.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table_ckpts.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table_ckpts.verticalHeader().setVisible(False)
        self.table_ckpts.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_ckpts.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table_ckpts.setMaximumHeight(140)
        ckpt_layout.addWidget(self.table_ckpts)
        btn_row = QHBoxLayout()
        self.btn_refresh = QPushButton('Refresh List')
        self.btn_refresh.clicked.connect(self._refresh_checkpoints_table)
        self.btn_load_selected = QPushButton('Load Selected Checkpoint')
        self.btn_load_selected.clicked.connect(self._load_selected_checkpoint)
        self.btn_load_custom = QPushButton('Browse Model File...')
        self.btn_load_custom.clicked.connect(self._load_custom_checkpoint)
        self.btn_save_current = QPushButton('Save Current Weights...')
        self.btn_save_current.clicked.connect(self._save_current_weights)
        btn_row.addWidget(self.btn_refresh)
        btn_row.addWidget(self.btn_load_selected)
        btn_row.addWidget(self.btn_load_custom)
        btn_row.addWidget(self.btn_save_current)
        ckpt_layout.addLayout(btn_row)
        layout.addWidget(ckpt_group)
        onnx_group = QGroupBox('Export Model to ONNX Format')
        onnx_layout = QVBoxLayout(onnx_group)
        export_row = QHBoxLayout()
        export_row.addWidget(QLabel('Output File:'))
        self.txt_onnx_path = QLineEdit('checkpoints/model.onnx')
        btn_browse_onnx = QPushButton('Browse...')
        btn_browse_onnx.clicked.connect(self._browse_onnx_output)
        export_row.addWidget(self.txt_onnx_path)
        export_row.addWidget(btn_browse_onnx)
        export_row.addWidget(QLabel('Precision:'))
        self.combo_onnx_prec = QComboBox()
        self.combo_onnx_prec.addItems(['FP32 (Standard)', 'FP16 (Half Precision)'])
        export_row.addWidget(self.combo_onnx_prec)
        self.btn_export_onnx = QPushButton('Export to ONNX')
        self.btn_export_onnx.clicked.connect(self._export_to_onnx)
        export_row.addWidget(self.btn_export_onnx)
        onnx_layout.addLayout(export_row)
        layout.addWidget(onnx_group)
        log_group = QGroupBox('Model Operations Log')
        log_layout = QVBoxLayout(log_group)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.document().setMaximumBlockCount(2000)
        log_layout.addWidget(self.txt_log)
        layout.addWidget(log_group, stretch=1)

    def _refresh_checkpoints_table(self):
        ckpt_dir = 'checkpoints'
        os.makedirs(ckpt_dir, exist_ok=True)
        files = [f for f in os.listdir(ckpt_dir) if f.endswith(('.pt', '.onnx'))]
        files.sort(key=lambda f: os.path.getmtime(os.path.join(ckpt_dir, f)), reverse=True)
        self.table_ckpts.setRowCount(len(files))
        for row, f in enumerate(files):
            full_path = os.path.join(ckpt_dir, f)
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(full_path)))
            fmt = 'ONNX (.onnx)' if f.endswith('.onnx') else 'PyTorch (.pt)'
            self.table_ckpts.setItem(row, 0, QTableWidgetItem(f))
            self.table_ckpts.setItem(row, 1, QTableWidgetItem(fmt))
            self.table_ckpts.setItem(row, 2, QTableWidgetItem(f'{size_mb:.2f}'))
            self.table_ckpts.setItem(row, 3, QTableWidgetItem(mtime))

    def _load_selected_checkpoint(self):
        row = self.table_ckpts.currentRow()
        if row < 0:
            QMessageBox.information(self, 'Select Checkpoint', 'Please select a checkpoint row in the table.')
            return
        filename = self.table_ckpts.item(row, 0).text()
        path = os.path.join('checkpoints', filename)
        self._load_checkpoint_file(path)

    def _load_custom_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open Model Checkpoint', 'checkpoints', 'Model Files (*.pt *.onnx);;ONNX Models (*.onnx);;PyTorch Weights (*.pt);;All Files (*)')
        if path:
            self._load_checkpoint_file(path)

    def _load_checkpoint_file(self, path: str):
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            loaded_model, meta = load_model(path, device=device)
            self.model = loaded_model
            self.active_checkpoint = path
            self.lbl_ckpt.setText(f'Loaded Weights: {path}')
            if getattr(self.model, 'is_onnx', False):
                self.lbl_arch.setText(f'Architecture: Ray-Knight ONNX ({self.model.count_parameters():,} params)')
                self.lbl_device.setText(f"Provider: {getattr(self.model, 'active_provider', 'ONNXRuntime')}")
            else:
                self.lbl_arch.setText(f'Architecture: Ray-Knight ({self.model.count_parameters():,} params)')
                dev_str = 'cuda (GPU)' if torch.cuda.is_available() else 'cpu'
                self.lbl_device.setText(f'Device: {dev_str} | Precision: Native FP16')
            step = meta.get('step', 0)
            step_str = f' | Step {step}' if step else ''
            self.txt_log.append(f'Successfully loaded checkpoint: {path} ({self.model.count_parameters():,} params{step_str})')
            self.checkpoint_loaded.emit(path)
            QMessageBox.information(self, 'Loaded', f'Successfully loaded weights from {os.path.basename(path)}.')
        except Exception as e:
            self.txt_log.append(f'Failed to load checkpoint: {e}')
            QMessageBox.critical(self, 'Error', f'Failed to load checkpoint: {e}')

    def _save_current_weights(self):
        if getattr(self.model, 'is_onnx', False):
            QMessageBox.information(self, 'ONNX Model Active', 'The currently loaded model is an ONNX inference model.\nONNX models are read-only and do not contain PyTorch optimizer or training state.\n\nTo save weights, please train or load a PyTorch (.pt) checkpoint.')
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save Weights As', 'checkpoints/custom_model.pt', 'PyTorch Weights (*.pt)')
        if path:
            try:
                os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
                save_checkpoint(path, self.model, config=self.model.get_config())
                self.txt_log.append(f'Saved weights to: {path}')
                self._refresh_checkpoints_table()
                QMessageBox.information(self, 'Saved', f'Saved weights to {os.path.basename(path)}.')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to save weights: {e}')

    def _browse_onnx_output(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save ONNX Model', self.txt_onnx_path.text(), 'ONNX Models (*.onnx)')
        if path:
            self.txt_onnx_path.setText(path)

    def _export_to_onnx(self):
        if getattr(self.model, 'is_onnx', False):
            QMessageBox.information(self, 'Already ONNX', 'The currently active model is already an ONNX model file.')
            return
        output_path = self.txt_onnx_path.text().strip()
        if not output_path:
            QMessageBox.warning(self, 'Missing Path', 'Please specify an output file path.')
            return
        half = 'FP16' in self.combo_onnx_prec.currentText()
        self.btn_export_onnx.setEnabled(False)
        self.export_worker = OnnxExportWorker(self.model, output_path=output_path, half_precision=half)
        self.export_worker.log.connect(self.txt_log.append)
        self.export_worker.finished.connect(self._on_onnx_export_finished)
        self.export_worker.start()

    def _on_onnx_export_finished(self, success: bool, msg: str):
        self.btn_export_onnx.setEnabled(True)
        if self.export_worker:
            self.export_worker.deleteLater()
            self.export_worker = None
        if success:
            QMessageBox.information(self, 'Export Successful', f'Model successfully exported to ONNX:\n{msg}')
        else:
            QMessageBox.critical(self, 'Export Failed', f'ONNX export encountered an error:\n{msg}')

    def close(self):
        if self.export_worker and self.export_worker.isRunning():
            self.export_worker.wait(1000)
            self.export_worker = None
        super().close()
