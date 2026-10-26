import os
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox, QSpinBox, QTextEdit, QProgressBar, QCheckBox, QLineEdit, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView
from chess_ai.dataset_generator import StockfishDatasetGenerator
from chess_ai.stockfish_runner import find_stockfish_binary

class StockfishGenerationWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(int)
    log = pyqtSignal(str)

    def __init__(self, num_games: int=20, movetime_ms: int=100, random_opening_plies: int=4, output_parquet: str='dataset/stockfish_selfplay.parquet', append: bool=True, stockfish_path: str | None=None, num_workers: int=4, threads_per_engine: int=1, randomize_skills: bool=True, min_skill: int=15, max_skill: int=20):
        super().__init__()
        self.num_games = num_games
        self.movetime_ms = movetime_ms
        self.random_opening_plies = random_opening_plies
        self.output_parquet = output_parquet
        self.append = append
        self.stockfish_path = find_stockfish_binary(stockfish_path)
        self.num_workers = max(1, num_workers)
        self.threads_per_engine = max(1, threads_per_engine)
        self.randomize_skills = randomize_skills
        self.min_skill = min_skill
        self.max_skill = max_skill
        self.running = True
        self.generator = StockfishDatasetGenerator(stockfish_path=self.stockfish_path, movetime_ms=self.movetime_ms, random_opening_plies=self.random_opening_plies, threads_per_engine=self.threads_per_engine, randomize_skills=self.randomize_skills, min_skill=self.min_skill, max_skill=self.max_skill)

    def run(self):
        skill_info = f'Randomized ({self.min_skill}-{self.max_skill}, White != Black)' if self.randomize_skills else 'Fixed Level 20 (Max Strength)'
        self.log.emit(f'Starting Stockfish self-play: generating {self.num_games} games using {self.num_workers} parallel CPU workers...')
        self.log.emit(f'Target dataset: {self.output_parquet} (append={self.append}, skills={skill_info}, movetime={self.movetime_ms}ms, opening_plies={self.random_opening_plies})')

        def should_stop():
            return not self.running

        def on_game(idx, total, record):
            moves_count = len(record['moves_uci'])
            msg = f"Game {idx}/{total} complete ({record['winner']}, {record['end_type']}, {moves_count} plies) | {record['source']}"
            self.progress.emit(idx, total, msg)
            self.log.emit(msg)
        try:
            count = self.generator.generate_games(num_games=self.num_games, output_parquet=self.output_parquet, append=self.append, num_workers=self.num_workers, should_stop=should_stop, on_game_complete=on_game)
            if self.running:
                self.log.emit(f'Dataset generation complete! Added {count} games to {self.output_parquet}.')
            else:
                self.log.emit(f'Generation stopped by user. {count} games safely preserved in {self.output_parquet}.')
            self.finished.emit(count)
        except Exception as e:
            self.log.emit(f'Stockfish generator error: {e}')
            self.finished.emit(0)

    def stop(self):
        self.running = False

class DatasetTab(QWidget):
    dataset_generated = pyqtSignal(str)
    stockfish_path_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.gen_worker: StockfishGenerationWorker | None = None
        self.manually_stopped = False
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        status_group = QGroupBox('Dataset Generation Status & Telemetry')
        status_layout = QVBoxLayout(status_group)
        self.lbl_status = QLabel('Generator State: Idle')
        self.lbl_status.setStyleSheet('font-weight: bold; font-size: 14px; color: #63b3ed;')
        status_layout.addWidget(self.lbl_status)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)
        self.lbl_metrics = QLabel('Games: 0 | Stockfish Skill: 20 (Max) | Move Time: 100 ms')
        status_layout.addWidget(self.lbl_metrics)
        layout.addWidget(status_group)
        sf_group = QGroupBox('Stockfish Self-Play Configuration')
        sf_layout = QVBoxLayout(sf_group)
        sf_row1 = QHBoxLayout()
        sf_row1.addWidget(QLabel('Games to Generate:'))
        self.spin_gen_games = QSpinBox()
        self.spin_gen_games.setRange(1, 50000)
        self.spin_gen_games.setSingleStep(10)
        self.spin_gen_games.setValue(20)
        self.spin_gen_games.setToolTip('Number of self-play games Stockfish will play against itself')
        sf_row1.addWidget(self.spin_gen_games)
        sf_row1.addWidget(QLabel('Move Time (ms):'))
        self.spin_gen_movetime = QSpinBox()
        self.spin_gen_movetime.setRange(20, 5000)
        self.spin_gen_movetime.setSingleStep(50)
        self.spin_gen_movetime.setValue(100)
        self.spin_gen_movetime.setToolTip('Stockfish thinking time per move in milliseconds')
        sf_row1.addWidget(self.spin_gen_movetime)
        sf_row1.addWidget(QLabel('Opening Plies:'))
        self.spin_gen_opening_plies = QSpinBox()
        self.spin_gen_opening_plies.setRange(0, 16)
        self.spin_gen_opening_plies.setValue(4)
        self.spin_gen_opening_plies.setToolTip('Number of opening random plies to create diverse opening variations')
        sf_row1.addWidget(self.spin_gen_opening_plies)
        sf_row1.addWidget(QLabel('CPU Workers:'))
        self.spin_gen_workers = QSpinBox()
        cpu_max = os.cpu_count() or 4
        self.spin_gen_workers.setRange(1, max(1, cpu_max))
        self.spin_gen_workers.setValue(min(4, max(1, cpu_max)))
        self.spin_gen_workers.setToolTip(f'Number of parallel CPU worker threads generating games simultaneously (1-{cpu_max}). Utilizing multiple cores accelerates dataset generation.')
        sf_row1.addWidget(self.spin_gen_workers)
        sf_layout.addLayout(sf_row1)
        sf_row2 = QHBoxLayout()
        self.chk_gen_random_skill = QCheckBox('Randomize Skill (White ≠ Black)')
        self.chk_gen_random_skill.setChecked(True)
        self.chk_gen_random_skill.setToolTip('When checked, White and Black are randomly assigned different Stockfish skill levels for each game.')
        sf_row2.addWidget(self.chk_gen_random_skill)
        sf_row2.addWidget(QLabel('Min Skill:'))
        self.spin_gen_min_skill = QSpinBox()
        self.spin_gen_min_skill.setRange(0, 20)
        self.spin_gen_min_skill.setValue(15)
        self.spin_gen_min_skill.setToolTip('Minimum Stockfish skill level (0-20)')
        sf_row2.addWidget(self.spin_gen_min_skill)
        sf_row2.addWidget(QLabel('Max Skill:'))
        self.spin_gen_max_skill = QSpinBox()
        self.spin_gen_max_skill.setRange(0, 20)
        self.spin_gen_max_skill.setValue(20)
        self.spin_gen_max_skill.setToolTip('Maximum Stockfish skill level (0-20)')
        sf_row2.addWidget(self.spin_gen_max_skill)
        self.lbl_sf_strength = QLabel('Stockfish Strength: Randomized Levels 15-20 (White ≠ Black)')
        self.lbl_sf_strength.setStyleSheet('font-weight: bold; color: #48bb78;')
        sf_row2.addWidget(self.lbl_sf_strength)
        sf_row2.addStretch(1)
        sf_layout.addLayout(sf_row2)
        self.chk_gen_random_skill.toggled.connect(self._update_strength_display)
        self.spin_gen_min_skill.valueChanged.connect(self._update_strength_display)
        self.spin_gen_max_skill.valueChanged.connect(self._update_strength_display)
        sf_row_bin = QHBoxLayout()
        sf_row_bin.addWidget(QLabel('Stockfish Executable:'))
        self.edit_stockfish_path = QLineEdit(find_stockfish_binary())
        self.edit_stockfish_path.textChanged.connect(self._on_stockfish_text_changed)
        sf_row_bin.addWidget(self.edit_stockfish_path)
        self.btn_browse_stockfish = QPushButton('Browse...')
        self.btn_browse_stockfish.clicked.connect(self._browse_stockfish_path)
        sf_row_bin.addWidget(self.btn_browse_stockfish)
        sf_layout.addLayout(sf_row_bin)
        sf_row3 = QHBoxLayout()
        sf_row3.addWidget(QLabel('Output Parquet:'))
        self.edit_gen_output = QLineEdit('dataset/stockfish_selfplay.parquet')
        sf_row3.addWidget(self.edit_gen_output)
        self.btn_browse_output = QPushButton('Browse...')
        self.btn_browse_output.clicked.connect(self._browse_gen_output)
        sf_row3.addWidget(self.btn_browse_output)
        self.chk_gen_append = QCheckBox('Append to existing')
        self.chk_gen_append.setChecked(True)
        sf_row3.addWidget(self.chk_gen_append)
        self.chk_gen_autoselect = QCheckBox('Auto-select in Training tab when complete')
        self.chk_gen_autoselect.setChecked(True)
        sf_row3.addWidget(self.chk_gen_autoselect)
        sf_layout.addLayout(sf_row3)
        layout.addWidget(sf_group)
        inv_group = QGroupBox('Dataset Inventory')
        inv_layout = QVBoxLayout(inv_group)
        inv_top = QHBoxLayout()
        self.lbl_inv_summary = QLabel('Discovered Parquet Datasets:')
        inv_top.addWidget(self.lbl_inv_summary)
        self.btn_refresh_inv = QPushButton('Refresh List')
        self.btn_refresh_inv.clicked.connect(self._refresh_dataset_inventory)
        inv_top.addWidget(self.btn_refresh_inv)
        inv_top.addStretch(1)
        inv_layout.addLayout(inv_top)
        self.table_datasets = QTableWidget(0, 2)
        self.table_datasets.setHorizontalHeaderLabels(['Dataset Path', 'File Size'])
        self.table_datasets.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table_datasets.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table_datasets.setMaximumHeight(110)
        inv_layout.addWidget(self.table_datasets)
        layout.addWidget(inv_group)
        ctrl_group = QGroupBox('Actions')
        ctrl_layout = QHBoxLayout(ctrl_group)
        self.btn_generate = QPushButton('Generate Stockfish Dataset')
        self.btn_generate.clicked.connect(self._start_generation)
        ctrl_layout.addWidget(self.btn_generate)
        self.btn_stop = QPushButton('Stop')
        self.btn_stop.setObjectName('dangerButton')
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_generation)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addStretch(1)
        layout.addWidget(ctrl_group)
        log_group = QGroupBox('Dataset Generation Log')
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
        self._refresh_dataset_inventory()

    def _refresh_dataset_inventory(self):
        self.table_datasets.setRowCount(0)
        if not os.path.exists('dataset'):
            return
        files = [f for f in os.listdir('dataset') if f.endswith('.parquet')]
        files.sort()
        self.table_datasets.setRowCount(len(files))
        for row, fname in enumerate(files):
            fpath = os.path.join('dataset', fname).replace('\\', '/')
            try:
                size_mb = os.path.getsize(fpath) / (1024 * 1024)
                size_str = f'{size_mb / 1024:.2f} GB' if size_mb >= 1024 else f'{size_mb:.1f} MB'
            except Exception:
                size_str = 'Unknown'
            item_path = QTableWidgetItem(fpath)
            item_path.setFlags(item_path.flags() ^ Qt.ItemFlag.ItemIsEditable)
            item_size = QTableWidgetItem(size_str)
            item_size.setFlags(item_size.flags() ^ Qt.ItemFlag.ItemIsEditable)
            self.table_datasets.setItem(row, 0, item_path)
            self.table_datasets.setItem(row, 1, item_size)
        self.lbl_inv_summary.setText(f'Discovered Parquet Datasets ({len(files)} found):')

    def _update_strength_display(self):
        if self.chk_gen_random_skill.isChecked():
            min_s = self.spin_gen_min_skill.value()
            max_s = self.spin_gen_max_skill.value()
            if min_s > max_s:
                min_s, max_s = (max_s, min_s)
            self.spin_gen_min_skill.setEnabled(True)
            self.spin_gen_max_skill.setEnabled(True)
            self.lbl_sf_strength.setText(f'Strength: Randomized Levels {min_s}-{max_s} (White ≠ Black)')
            self.lbl_sf_strength.setStyleSheet('font-weight: bold; color: #48bb78;')
        else:
            self.spin_gen_min_skill.setEnabled(False)
            self.spin_gen_max_skill.setEnabled(False)
            self.lbl_sf_strength.setText('Strength: Skill Level 20 (Max)')
            self.lbl_sf_strength.setStyleSheet('font-weight: bold; color: #63b3ed;')

    def _browse_stockfish_path(self):
        curr_text = self.edit_stockfish_path.text().strip()
        start_dir = os.path.dirname(curr_text) if curr_text else '.'
        path, _ = QFileDialog.getOpenFileName(self, 'Select Stockfish Executable', start_dir, 'Executables (*.exe);;All Files (*)')
        if path:
            self.edit_stockfish_path.setText(path.replace('\\', '/'))

    def _on_stockfish_text_changed(self, text: str):
        self.stockfish_path_changed.emit(text.strip())

    def set_stockfish_path(self, path: str):
        if self.edit_stockfish_path.text().strip() != path.strip():
            self.edit_stockfish_path.blockSignals(True)
            self.edit_stockfish_path.setText(path)
            self.edit_stockfish_path.blockSignals(False)

    def _browse_gen_output(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Select Output Parquet Path', self.edit_gen_output.text() or 'dataset/stockfish_selfplay.parquet', 'Parquet Files (*.parquet);;All Files (*)')
        if path:
            self.edit_gen_output.setText(path.replace('\\', '/'))

    def _start_generation(self):
        self.manually_stopped = False
        self.btn_generate.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText('Generator State: Generating Stockfish Self-Play Dataset')
        num_games = self.spin_gen_games.value()
        movetime = self.spin_gen_movetime.value()
        opening_plies = self.spin_gen_opening_plies.value()
        num_workers = self.spin_gen_workers.value()
        random_skill = self.chk_gen_random_skill.isChecked()
        min_skill = self.spin_gen_min_skill.value()
        max_skill = self.spin_gen_max_skill.value()
        output_parquet = self.edit_gen_output.text().strip() or 'dataset/stockfish_selfplay.parquet'
        append = self.chk_gen_append.isChecked()
        skill_desc = f'Random {min_skill}-{max_skill}' if random_skill else '20 (Max)'
        self.progress_bar.setRange(0, num_games)
        self.progress_bar.setValue(0)
        self.lbl_metrics.setText(f'Games: 0/{num_games} | Workers: {num_workers} | Skill: {skill_desc} | Move Time: {movetime} ms')
        self.gen_worker = StockfishGenerationWorker(num_games=num_games, movetime_ms=movetime, random_opening_plies=opening_plies, output_parquet=output_parquet, append=append, stockfish_path=self.edit_stockfish_path.text().strip(), num_workers=num_workers, randomize_skills=random_skill, min_skill=min_skill, max_skill=max_skill)
        self.gen_worker.progress.connect(self._on_gen_progress)
        self.gen_worker.log.connect(self._on_log)
        self.gen_worker.finished.connect(self._on_gen_finished)
        self.gen_worker.start()

    def _on_gen_progress(self, current: int, total: int, desc: str):
        if self.manually_stopped:
            return
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
        self.lbl_status.setText(f'Generator State: {desc}')
        self.lbl_metrics.setText(f'Stockfish Self-Play Progress: {current}/{total} games generated')

    def _on_gen_finished(self, count: int):
        self.btn_generate.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self.manually_stopped:
            self.lbl_status.setText('Generator State: Idle')
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.lbl_metrics.setText('Stockfish Self-Play Progress: Idle (Ready)')
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)
            self.lbl_status.setText('Generator State: Idle (Dataset Generation Finished)')
        if self.gen_worker:
            self.gen_worker.deleteLater()
            self.gen_worker = None
        self._refresh_dataset_inventory()
        out_path = self.edit_gen_output.text().strip() or 'dataset/stockfish_selfplay.parquet'
        if os.path.exists(out_path):
            self.dataset_generated.emit(out_path)

    def _stop_generation(self):
        self.manually_stopped = True
        self.btn_stop.setEnabled(False)
        if self.gen_worker:
            self.gen_worker.stop()
        self.lbl_status.setText('Generator State: Idle')
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.lbl_metrics.setText('Stockfish Self-Play Progress: Idle (Ready)')
        self.txt_log.append('Stopping active dataset generation (manual stop)...')

    def _on_log(self, text: str):
        self.txt_log.append(text)

    def close(self):
        if self.gen_worker and self.gen_worker.isRunning():
            self.gen_worker.stop()
            self.gen_worker.wait(1000)
            self.gen_worker = None
        super().close()
