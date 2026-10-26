import os
import torch
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox, QSpinBox, QTextEdit, QProgressBar, QCheckBox, QComboBox, QFileDialog
from chess_ai.model import ChessNet, MODEL_PRESETS
from chess_ai.trainer import Trainer
from chess_ai.gui.dataset_tab import StockfishGenerationWorker

class SupervisedWorker(QThread):
    progress = pyqtSignal(int, float, float, float)
    finished = pyqtSignal()
    log = pyqtSignal(str)

    def __init__(self, model: ChessNet, parquet_path: str='dataset/dataset.parquet', max_steps: int=200, steps: int | None=None, time_limit_sec: float | None=None, scheduler: str='wsd', batch_size: int=256, min_elo: int | None=2000, max_elo: int | None=None, lr: float=0.001, num_workers: int=4, save_dir: str='checkpoints', resume: bool=False):
        super().__init__()
        self.model = model
        self.parquet_path = parquet_path
        self.max_steps = max_steps if steps is None else steps
        self.steps = steps if steps is not None else max_steps
        self.time_limit_sec = time_limit_sec
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.min_elo = min_elo
        self.max_elo = max_elo
        self.lr = lr
        self.num_workers = num_workers
        self.save_dir = save_dir
        self.resume = resume
        self.running = True
        self.trainer: Trainer | None = None

    def run(self):
        elo_desc = f"min={self.min_elo}, max={self.max_elo or 'unlimited'}"
        self.log.emit(f"Starting supervised training on dataset '{self.parquet_path}' (Elo filter: {elo_desc}, workers: {self.num_workers}, resume={self.resume})...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            resume_checkpoint = None
            if isinstance(self.resume, str) and self.resume:
                resume_checkpoint = self.resume
            elif self.resume:
                default_ckpt = os.path.join(self.save_dir, 'last.pt')
                if os.path.exists(default_ckpt):
                    resume_checkpoint = default_ckpt
                else:
                    self.log.emit(f"Resume requested, but '{default_ckpt}' does not exist. Starting training from current model weights.")
            self.trainer = Trainer(model=self.model, parquet_path=self.parquet_path, save_dir=self.save_dir, batch_size=self.batch_size, lr=self.lr, min_elo=self.min_elo, max_elo=self.max_elo, num_workers=self.num_workers, device=device, resume_checkpoint=resume_checkpoint)

            def on_progress(step, max_st, loss, lp, lv, top1, cur_lr, nps):
                self.progress.emit(step, loss, lp, top1)

            def on_log(text):
                self.log.emit(text)

            def should_stop():
                return not self.running
            self.model = self.trainer.train(steps=self.steps, max_steps=self.max_steps, time_limit_sec=self.time_limit_sec, scheduler_type=self.scheduler, resume_checkpoint=resume_checkpoint, on_progress=on_progress, on_log=on_log, should_stop=should_stop)
            if self.running:
                self.log.emit('Supervised training session complete.')
            else:
                self.log.emit('Supervised training session stopped.')
        except Exception as e:
            self.log.emit(f'Training error: {e}')
        finally:
            if self.trainer is not None:
                try:
                    self.trainer.close()
                except Exception:
                    pass
                self.trainer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.finished.emit()

    def stop(self):
        self.running = False

class TrainTab(QWidget):
    model_updated = pyqtSignal(ChessNet, str)

    def __init__(self, model: ChessNet | None=None, parent=None):
        super().__init__(parent)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model or ChessNet(**MODEL_PRESETS['fast']).to(self.device)
        self.worker: SupervisedWorker | None = None
        self.manually_stopped = False
        self._session_start_step: int | None = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        status_group = QGroupBox('Training Status & Telemetry')
        status_layout = QVBoxLayout(status_group)
        self.lbl_status = QLabel('Engine State: Idle')
        self.lbl_status.setStyleSheet('font-weight: bold; font-size: 14px; color: #63b3ed;')
        status_layout.addWidget(self.lbl_status)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)
        self.lbl_metrics = QLabel('Step: 0 | Total Loss: - | Policy Loss: - | Top-1 Accuracy: -')
        status_layout.addWidget(self.lbl_metrics)
        layout.addWidget(status_group)
        cfg_group = QGroupBox('Supervised Training Configuration')
        cfg_layout = QVBoxLayout(cfg_group)
        ds_row = QHBoxLayout()
        ds_row.addWidget(QLabel('Dataset:'))
        self.combo_dataset = QComboBox()
        self.combo_dataset.setMinimumWidth(260)
        self.combo_dataset.currentIndexChanged.connect(self._on_dataset_selection_changed)
        ds_row.addWidget(self.combo_dataset)
        self.btn_browse_ds = QPushButton('Browse...')
        self.btn_browse_ds.clicked.connect(self._browse_dataset)
        ds_row.addWidget(self.btn_browse_ds)
        self.btn_refresh_ds = QPushButton('Refresh')
        self.btn_refresh_ds.clicked.connect(lambda: self._refresh_dataset_list())
        ds_row.addWidget(self.btn_refresh_ds)
        self.lbl_dataset_info = QLabel('Size: Detecting...')
        self.lbl_dataset_info.setStyleSheet('color: #a0aec0; font-size: 11px;')
        ds_row.addWidget(self.lbl_dataset_info, stretch=1)
        cfg_layout.addLayout(ds_row)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel('Model Preset:'))
        self.combo_preset = QComboBox()
        self.combo_preset.addItems(['Fast (10 blocks x 128 filters)', 'Full (32 blocks x 256 filters)'])
        row1.addWidget(self.combo_preset)
        row1.addWidget(QLabel('LR Schedule:'))
        self.combo_sched = QComboBox()
        self.combo_sched.addItems(['WSD', 'Cosine', 'OneCycle'])
        row1.addWidget(self.combo_sched)
        row1.addWidget(QLabel('Batch Size:'))
        self.spin_batch_size = QSpinBox()
        self.spin_batch_size.setRange(8, 1024)
        self.spin_batch_size.setSingleStep(64)
        self.spin_batch_size.setValue(256)
        row1.addWidget(self.spin_batch_size)
        row1.addWidget(QLabel('Steps:'))
        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(20, 500000)
        self.spin_steps.setSingleStep(500)
        self.spin_steps.setValue(1000)
        row1.addWidget(self.spin_steps)
        row1.addWidget(QLabel('Time Limit (min):'))
        self.spin_time_limit = QSpinBox()
        self.spin_time_limit.setRange(0, 2880)
        self.spin_time_limit.setSingleStep(15)
        self.spin_time_limit.setValue(0)
        self.spin_time_limit.setToolTip('0 = Unlimited (run for the full number of steps)')
        row1.addWidget(self.spin_time_limit)
        row1.addWidget(QLabel('Data Workers:'))
        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(0, 16)
        self.spin_workers.setValue(4)
        row1.addWidget(self.spin_workers)
        cfg_layout.addLayout(row1)
        filter_layout = QHBoxLayout()
        self.chk_elo = QCheckBox('Enable Elo Filter')
        self.chk_elo.setChecked(True)
        self.chk_elo.toggled.connect(self._on_toggle_elo_filter)
        filter_layout.addWidget(self.chk_elo)
        filter_layout.addWidget(QLabel('Min Elo:'))
        self.spin_min_elo = QSpinBox()
        self.spin_min_elo.setRange(0, 3500)
        self.spin_min_elo.setSingleStep(50)
        self.spin_min_elo.setValue(2000)
        filter_layout.addWidget(self.spin_min_elo)
        filter_layout.addWidget(QLabel('Max Elo:'))
        self.spin_max_elo = QSpinBox()
        self.spin_max_elo.setRange(500, 4000)
        self.spin_max_elo.setSingleStep(50)
        self.spin_max_elo.setValue(3500)
        filter_layout.addWidget(self.spin_max_elo)
        self.chk_resume = QCheckBox('Resume from checkpoints/last.pt if present')
        self.chk_resume.setChecked(False)
        filter_layout.addWidget(self.chk_resume)
        cfg_layout.addLayout(filter_layout)
        layout.addWidget(cfg_group)
        ctrl_group = QGroupBox('Actions')
        ctrl_layout = QHBoxLayout(ctrl_group)
        self.btn_train = QPushButton('Start Supervised Training')
        self.btn_train.clicked.connect(self._start_supervised)
        ctrl_layout.addWidget(self.btn_train)
        self.btn_stop = QPushButton('Stop')
        self.btn_stop.setObjectName('dangerButton')
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_training)
        ctrl_layout.addWidget(self.btn_stop)
        self.chk_shutdown = QCheckBox('Shut down PC when complete')
        self.chk_shutdown.setToolTip('Automatically shut down the computer upon training completion (not triggered by Stop button).')
        ctrl_layout.addWidget(self.chk_shutdown)
        self.btn_abort_shutdown = QPushButton('Abort Shutdown')
        self.btn_abort_shutdown.setObjectName('dangerButton')
        self.btn_abort_shutdown.setVisible(False)
        self.btn_abort_shutdown.clicked.connect(self._abort_pc_shutdown)
        ctrl_layout.addWidget(self.btn_abort_shutdown)
        ctrl_layout.addStretch(1)
        layout.addWidget(ctrl_group)
        log_group = QGroupBox('Training Output Log')
        log_layout = QVBoxLayout(log_group)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.document().setMaximumBlockCount(2000)
        log_layout.addWidget(self.txt_log)
        log_btn_row = QHBoxLayout()
        log_btn_row.addStretch(1)
        self.btn_clear_log = QPushButton('Clear Log')
        self.btn_clear_log.clicked.connect(self.txt_log.clear)
        log_btn_row.addWidget(self.btn_clear_log)
        log_layout.addLayout(log_btn_row)
        layout.addWidget(log_group, stretch=1)
        self._refresh_dataset_list()

    def on_dataset_generated(self, parquet_path: str):
        self._refresh_dataset_list(select_path=parquet_path)
        self.txt_log.append(f'Auto-selected newly generated dataset: {parquet_path}')

    def _refresh_dataset_list(self, select_path: str | None=None):
        current_selection = select_path or self.combo_dataset.currentText()
        self.combo_dataset.blockSignals(True)
        self.combo_dataset.clear()
        discovered = []
        if os.path.exists('dataset'):
            for fname in os.listdir('dataset'):
                if fname.endswith('.parquet'):
                    discovered.append(os.path.join('dataset', fname).replace('\\', '/'))
        if 'dataset/dataset.parquet' not in discovered and os.path.exists('dataset/dataset.parquet'):
            discovered.insert(0, 'dataset/dataset.parquet')
        for ds_path in discovered:
            self.combo_dataset.addItem(ds_path)
        if current_selection and current_selection not in discovered and os.path.exists(current_selection):
            self.combo_dataset.addItem(current_selection)
        target = select_path or current_selection
        idx = self.combo_dataset.findText(target)
        if idx >= 0:
            self.combo_dataset.setCurrentIndex(idx)
        elif self.combo_dataset.count() > 0:
            self.combo_dataset.setCurrentIndex(0)
        self.combo_dataset.blockSignals(False)
        self._update_dataset_info()

    def _on_dataset_selection_changed(self):
        self._update_dataset_info()

    def _update_dataset_info(self):
        path = self.get_selected_dataset()
        if not path or not os.path.exists(path):
            self.lbl_dataset_info.setText('File not found')
            return
        try:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            if size_mb >= 1024:
                self.lbl_dataset_info.setText(f'Size: {size_mb / 1024:.2f} GB')
            else:
                self.lbl_dataset_info.setText(f'Size: {size_mb:.1f} MB')
        except Exception:
            self.lbl_dataset_info.setText('Detected')

    def get_selected_dataset(self) -> str:
        return self.combo_dataset.currentText().strip() or 'dataset/dataset.parquet'

    def _browse_dataset(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select Training Dataset', 'dataset', 'Parquet Files (*.parquet);;All Files (*)')
        if path:
            path = path.replace('\\', '/')
            if self.combo_dataset.findText(path) < 0:
                self.combo_dataset.addItem(path)
            self.combo_dataset.setCurrentText(path)
            self._update_dataset_info()

    def _on_toggle_elo_filter(self, checked: bool):
        self.spin_min_elo.setEnabled(checked)
        self.spin_max_elo.setEnabled(checked)

    def _start_supervised(self):
        self.manually_stopped = False
        self._session_start_step: int | None = None
        self._session_total_steps: int = self.spin_steps.value()
        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText('Engine State: Training (Supervised Pretraining)')
        self.progress_bar.setRange(0, self._session_total_steps)
        self.progress_bar.setValue(0)
        self.lbl_metrics.setText('Step: 0 | Total Loss: - | Policy Loss: - | Top-1 Accuracy: -')
        min_elo = self.spin_min_elo.value() if self.chk_elo.isChecked() else None
        max_elo = self.spin_max_elo.value() if self.chk_elo.isChecked() else None
        preset_key = 'fast' if 'Fast' in self.combo_preset.currentText() else 'full'
        target_cfg = MODEL_PRESETS[preset_key]
        import copy
        if getattr(self.model, 'is_onnx', False):
            train_model = ChessNet(**target_cfg).to(self.device).float()
            self.txt_log.append(f'[Notice] Active model is an ONNX inference graph. Initialized a fresh trainable PyTorch {preset_key.upper()} model for this training session.')
        elif self.model.num_blocks != target_cfg['num_blocks'] or self.model.num_filters != target_cfg['num_filters']:
            train_model = ChessNet(**target_cfg).to(self.device).float()
        else:
            train_model = copy.deepcopy(self.model).to(self.device).float()
        time_limit_sec = self.spin_time_limit.value() * 60.0 if self.spin_time_limit.value() > 0 else None
        lr = 0.001 if preset_key == 'fast' else 0.0003
        dataset_path = self.get_selected_dataset()
        self.worker = SupervisedWorker(train_model, parquet_path=dataset_path, max_steps=self.spin_steps.value(), time_limit_sec=time_limit_sec, scheduler=self.combo_sched.currentText().lower(), batch_size=self.spin_batch_size.value(), min_elo=min_elo, max_elo=max_elo, lr=lr, num_workers=self.spin_workers.value(), resume=self.chk_resume.isChecked())
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._on_log)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _stop_training(self):
        self.manually_stopped = True
        self._session_start_step = None
        self.btn_stop.setEnabled(False)
        if self.worker:
            self.worker.stop()
        self.lbl_status.setText('Engine State: Idle')
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.lbl_metrics.setText('Step: 0 | Total Loss: - | Policy Loss: - | Top-1 Accuracy: -')
        self.txt_log.append('Stopping active training session (manual stop)...')

    def _on_progress(self, step: int, loss: float, p_loss: float, top1: float):
        if self.manually_stopped:
            return
        max_val = self.progress_bar.maximum()
        if step <= max_val:
            self.progress_bar.setValue(step)
        else:
            if self._session_start_step is None:
                self._session_start_step = step
            session_step = step - self._session_start_step + 1
            self.progress_bar.setValue(min(session_step, max_val))
        self.lbl_metrics.setText(f'Step: {step} | Total Loss: {loss:.4f} | Policy Loss: {p_loss:.4f} | Top-1 Accuracy: {top1 * 100:.1f}%')

    def _on_log(self, text: str):
        self.txt_log.append(text)

    def _on_finished(self):
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self.manually_stopped:
            self.lbl_status.setText('Engine State: Idle')
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.lbl_metrics.setText('Step: 0 | Total Loss: - | Policy Loss: - | Top-1 Accuracy: -')
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)
            self.lbl_status.setText('Engine State: Idle (Finished)')
        if self.worker and self.worker.model:
            self.model = self.worker.model
            self.model.eval()
            if self.device == 'cuda':
                self.model.half()
            last_ckpt = os.path.join(self.worker.save_dir, 'last.pt')
            best_ckpt = os.path.join(self.worker.save_dir, 'best_model.pt')
            saved_path = last_ckpt if os.path.exists(last_ckpt) else best_ckpt if os.path.exists(best_ckpt) else 'Active Memory'
            self.model_updated.emit(self.model, saved_path)
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        if not self.manually_stopped and self.chk_shutdown.isChecked():
            self._trigger_pc_shutdown()

    def close(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1000)
            self.worker = None
        super().close()

    def _trigger_pc_shutdown(self):
        self.txt_log.append('Training completed normally. Initiating PC shutdown in 60 seconds...')
        self.btn_abort_shutdown.setVisible(True)
        try:
            import subprocess
            subprocess.run(['shutdown', '/s', '/t', '60', '/c', 'Chess AI training complete. Shutting down system.'], check=False)
            self.txt_log.append("Shutdown command scheduled. Click 'Abort Shutdown' to cancel.")
        except Exception as e:
            self.txt_log.append(f'Failed to initiate shutdown: {e}')

    def _abort_pc_shutdown(self):
        try:
            import subprocess
            subprocess.run(['shutdown', '/a'], check=False)
            self.btn_abort_shutdown.setVisible(False)
            self.txt_log.append('PC shutdown successfully aborted.')
        except Exception as e:
            self.txt_log.append(f'Failed to cancel shutdown: {e}')
