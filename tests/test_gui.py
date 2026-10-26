import os
import pytest
from PyQt6.QtWidgets import QApplication
from chess_ai.gui.main_window import MainWindow
from chess_ai.gui.dataset_tab import StockfishGenerationWorker, DatasetTab
from chess_ai.gui.train_tab import TrainTab
from chess_ai.model import ChessNet

@pytest.fixture(scope='session')
def qapp():
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app

def test_main_window_instantiation(qapp):
    model = ChessNet(num_blocks=2, num_filters=64)
    win = MainWindow()
    win.model = model
    assert win.tabs.count() == 5
    assert win.tabs.tabText(0) == 'Play & Analysis'
    assert win.tabs.tabText(1) == 'Model vs Stockfish Match'
    assert win.tabs.tabText(2) == 'Training'
    assert win.tabs.tabText(3) == 'Dataset Generation'
    assert win.tabs.tabText(4) == 'Model Management'
    assert isinstance(win.train_tab, TrainTab)
    assert isinstance(win.dataset_tab, DatasetTab)

def test_train_tab_shutdown_logic(qapp, monkeypatch):
    win = MainWindow()
    train_tab = win.train_tab
    shutdown_called = []
    monkeypatch.setattr(train_tab, '_trigger_pc_shutdown', lambda: shutdown_called.append(True))
    train_tab.chk_shutdown.setChecked(False)
    train_tab.manually_stopped = False
    train_tab._on_finished()
    assert len(shutdown_called) == 0
    train_tab.chk_shutdown.setChecked(True)
    train_tab.manually_stopped = True
    train_tab._on_finished()
    assert len(shutdown_called) == 0
    train_tab.chk_shutdown.setChecked(True)
    train_tab.manually_stopped = False
    train_tab._on_finished()
    assert len(shutdown_called) == 1

def test_train_tab_controls(qapp):
    win = MainWindow()
    train_tab = win.train_tab
    assert hasattr(train_tab, 'combo_dataset')
    assert hasattr(train_tab, 'btn_browse_ds')
    assert hasattr(train_tab, 'btn_refresh_ds')
    assert hasattr(train_tab, 'lbl_dataset_info')
    assert hasattr(train_tab, 'combo_preset')
    assert hasattr(train_tab, 'combo_sched')
    assert hasattr(train_tab, 'spin_batch_size')
    assert hasattr(train_tab, 'spin_steps')
    assert hasattr(train_tab, 'spin_time_limit')
    assert hasattr(train_tab, 'spin_workers')
    assert hasattr(train_tab, 'chk_elo')
    assert hasattr(train_tab, 'spin_min_elo')
    assert hasattr(train_tab, 'spin_max_elo')
    assert hasattr(train_tab, 'chk_resume')
    assert hasattr(train_tab, 'btn_train')
    assert hasattr(train_tab, 'btn_stop')
    assert hasattr(train_tab, 'chk_shutdown')
    assert hasattr(train_tab, 'txt_log')

def test_dataset_tab_controls(qapp):
    win = MainWindow()
    dataset_tab = win.dataset_tab
    assert hasattr(dataset_tab, 'edit_stockfish_path')
    assert hasattr(dataset_tab, 'btn_browse_stockfish')
    assert hasattr(dataset_tab, 'spin_gen_games')
    assert hasattr(dataset_tab, 'spin_gen_movetime')
    assert hasattr(dataset_tab, 'spin_gen_opening_plies')
    assert hasattr(dataset_tab, 'spin_gen_workers')
    assert hasattr(dataset_tab, 'chk_gen_random_skill')
    assert hasattr(dataset_tab, 'spin_gen_min_skill')
    assert hasattr(dataset_tab, 'spin_gen_max_skill')
    assert hasattr(dataset_tab, 'edit_gen_output')
    assert hasattr(dataset_tab, 'chk_gen_append')
    assert hasattr(dataset_tab, 'chk_gen_autoselect')
    assert hasattr(dataset_tab, 'btn_generate')
    assert hasattr(dataset_tab, 'btn_stop')
    assert hasattr(dataset_tab, 'table_datasets')
    assert hasattr(dataset_tab, 'btn_refresh_inv')
    assert hasattr(dataset_tab, 'txt_log')
    assert dataset_tab.spin_gen_games.value() == 20
    assert dataset_tab.spin_gen_movetime.value() == 100
    assert dataset_tab.spin_gen_opening_plies.value() == 4
    assert dataset_tab.chk_gen_append.isChecked() is True
    assert dataset_tab.chk_gen_autoselect.isChecked() is True
    assert dataset_tab.chk_gen_random_skill.isChecked() is True
    assert dataset_tab.spin_gen_min_skill.value() == 15
    assert dataset_tab.spin_gen_max_skill.value() == 20
    assert dataset_tab.spin_gen_workers.value() >= 1

def test_train_tab_dataset_selection(qapp, tmp_path):
    win = MainWindow()
    train_tab = win.train_tab
    dummy_ds = str(tmp_path / 'custom.parquet').replace('\\', '/')
    with open(dummy_ds, 'w') as f:
        f.write('test')
    train_tab._refresh_dataset_list(select_path=dummy_ds)
    assert train_tab.get_selected_dataset() == dummy_ds

def test_dataset_tab_start_generation_parameters(qapp, monkeypatch):
    win = MainWindow()
    dataset_tab = win.dataset_tab
    monkeypatch.setattr(StockfishGenerationWorker, 'start', lambda self: None)
    dataset_tab.spin_gen_games.setValue(15)
    dataset_tab.spin_gen_movetime.setValue(80)
    dataset_tab.spin_gen_opening_plies.setValue(6)
    dataset_tab.spin_gen_workers.setValue(3)
    dataset_tab.chk_gen_random_skill.setChecked(True)
    dataset_tab.spin_gen_min_skill.setValue(16)
    dataset_tab.spin_gen_max_skill.setValue(19)
    dataset_tab.edit_gen_output.setText('dataset/custom_sf.parquet')
    dataset_tab.chk_gen_append.setChecked(False)
    dataset_tab.edit_stockfish_path.setText('custom/path/stockfish.exe')
    dataset_tab._start_generation()
    worker = dataset_tab.gen_worker
    assert worker is not None
    assert worker.stockfish_path == 'custom/path/stockfish.exe'
    assert worker.num_games == 15
    assert worker.movetime_ms == 80
    assert worker.random_opening_plies == 6
    assert worker.num_workers == 3
    assert worker.randomize_skills is True
    assert worker.min_skill == 16
    assert worker.max_skill == 19
    assert worker.output_parquet == 'dataset/custom_sf.parquet'
    assert worker.append is False
    dataset_tab.chk_gen_random_skill.setChecked(False)
    assert dataset_tab.spin_gen_min_skill.isEnabled() is False
    assert dataset_tab.spin_gen_max_skill.isEnabled() is False
    assert 'Level 20 (Max)' in dataset_tab.lbl_sf_strength.text()
    assert worker.running is True
    dataset_tab._stop_generation()
    assert worker.running is False

def test_cross_tab_dataset_sync(qapp, tmp_path):
    win = MainWindow()
    dummy_ds = str(tmp_path / 'synced_dataset.parquet').replace('\\', '/')
    with open(dummy_ds, 'w') as f:
        f.write('sync test')
    win.dataset_tab.chk_gen_autoselect.setChecked(True)
    win.dataset_tab.dataset_generated.emit(dummy_ds)
    assert win.train_tab.get_selected_dataset() == dummy_ds

def test_stockfish_generation_worker_stop(qapp, tmp_path, monkeypatch):
    out_path = str(tmp_path / 'stopped_sf.parquet')
    worker = StockfishGenerationWorker(num_games=10, movetime_ms=30, random_opening_plies=2, output_parquet=out_path)
    worker.running = False
    monkeypatch.setattr(worker.generator, 'generate_games', lambda *args, **kwargs: 0)
    worker.run()
    assert worker.running is False

def test_supervised_worker_resume(qapp, tmp_path):
    from chess_ai.gui.train_tab import SupervisedWorker
    from chess_ai.model import ChessNet
    from chess_ai.dataset_generator import save_records_to_parquet
    dummy_ds = str(tmp_path / 'train.parquet')
    save_records_to_parquet([{'date': '2026.10.23', 'white_elo': 2500, 'black_elo': 2500, 'winner': 'white', 'moves_uci': ['e2e4', 'e7e5'], 'end_type': 'checkmate', 'source': 'test'}] * 10, dummy_ds, append=False)
    model = ChessNet(num_blocks=2, num_filters=32)
    worker = SupervisedWorker(model=model, parquet_path=dummy_ds, max_steps=1, save_dir=str(tmp_path / 'ckpts'), resume=False)
    logs = []
    worker.log.connect(lambda msg: logs.append(msg))
    worker.run()
    assert not any(('Training error' in msg for msg in logs))
    assert any(('Supervised training session complete' in msg for msg in logs))
    logs_resume = []
    worker_resume = SupervisedWorker(model=model, parquet_path=dummy_ds, max_steps=1, save_dir=str(tmp_path / 'ckpts_res'), resume=True)
    worker_resume.log.connect(lambda msg: logs_resume.append(msg))
    worker_resume.run()
    assert not any(('Training error' in msg for msg in logs_resume))
    assert any(('Resume requested, but' in msg for msg in logs_resume))
    assert any(('Supervised training session complete' in msg for msg in logs_resume))
    import torch
    from chess_ai.model import save_checkpoint
    save_dir_existing = str(tmp_path / 'ckpts_existing')
    os.makedirs(save_dir_existing, exist_ok=True)
    save_checkpoint(os.path.join(save_dir_existing, 'last.pt'), model, step=180020)
    logs_existing = []
    worker_existing = SupervisedWorker(model=model, parquet_path=dummy_ds, max_steps=2, save_dir=save_dir_existing, resume=True)
    worker_existing.log.connect(lambda msg: logs_existing.append(msg))
    worker_existing.run()
    assert not any(('Training error' in msg for msg in logs_existing))
    assert any(('Supervised training session complete' in msg for msg in logs_existing))
    saved_data = torch.load(os.path.join(save_dir_existing, 'last.pt'), map_location='cpu')
    assert saved_data['step'] == 180022, f"Expected step 180022 but got {saved_data['step']}"

def test_gui_main_window_autoload_onnx(qapp, monkeypatch):
    monkeypatch.setattr('chess_ai.gui.main_window.find_latest_checkpoint', lambda *args, **kwargs: 'checkpoints/model.onnx')
    win = MainWindow()
    assert getattr(win.model, 'is_onnx', False) is True
    assert win.model_tab.table_ckpts.rowCount() >= 1
    assert win.model_tab.table_ckpts.columnCount() == 4
    header_labels = [win.model_tab.table_ckpts.horizontalHeaderItem(i).text() for i in range(4)]
    assert 'Format' in header_labels
    assert 'Architecture: Ray-Knight ONNX' in win.model_tab.lbl_arch.text()

def test_gui_train_tab_with_active_onnx_model(qapp, monkeypatch):
    monkeypatch.setattr('chess_ai.gui.main_window.find_latest_checkpoint', lambda *args, **kwargs: 'checkpoints/model.onnx')
    win = MainWindow()
    train_tab = win.train_tab
    assert getattr(win.model, 'is_onnx', False) is True
    from chess_ai.gui.train_tab import SupervisedWorker
    monkeypatch.setattr(SupervisedWorker, 'start', lambda self: None)
    train_tab.get_selected_dataset = lambda: 'dummy.parquet'
    train_tab._start_supervised()
    assert 'Active model is an ONNX inference graph' in train_tab.txt_log.toPlainText()

def test_gui_train_tab_progress_bar_with_checkpoint_resume(qapp, monkeypatch):
    from chess_ai.gui.train_tab import TrainTab, SupervisedWorker
    train_tab = TrainTab()
    monkeypatch.setattr(SupervisedWorker, 'start', lambda self: None)
    train_tab.get_selected_dataset = lambda: 'dummy.parquet'
    train_tab.spin_steps.setValue(1000)
    train_tab._start_supervised()
    assert train_tab.progress_bar.maximum() == 1000
    assert train_tab.progress_bar.value() == 0
    train_tab._on_progress(180021, 1.5, 1.0, 0.5)
    assert train_tab.progress_bar.value() == 1
    assert 'Step: 180021' in train_tab.lbl_metrics.text()
    train_tab._on_progress(180520, 1.2, 0.8, 0.6)
    assert train_tab.progress_bar.value() == 500
    train_tab._on_progress(181020, 1.0, 0.6, 0.7)
    assert train_tab.progress_bar.value() == 1000
    assert 'Step: 181020' in train_tab.lbl_metrics.text()

def test_board_widget_zero_and_minimal_size_paint(qapp):
    from chess_ai.gui.board_widget import ChessBoardWidget
    from PyQt6.QtGui import QPaintEvent
    from PyQt6.QtCore import QRect
    bw = ChessBoardWidget()
    assert bw._piece_font.pointSize() > 0
    assert bw._coord_font.pointSize() > 0
    bw.resize(0, 0)
    bw.paintEvent(QPaintEvent(QRect(0, 0, 0, 0)))
    bw.resize(1, 1)
    bw.paintEvent(QPaintEvent(QRect(0, 0, 1, 1)))
    assert bw._piece_font.pointSize() >= 1
    assert bw._coord_font.pointSize() >= 1

def test_train_tab_stop_resets_state_and_counters(qapp):
    win = MainWindow()
    train_tab = win.train_tab
    train_tab.lbl_status.setText('Engine State: Training (Supervised Pretraining)')
    train_tab.progress_bar.setRange(0, 1000)
    train_tab.btn_train.setEnabled(False)
    train_tab.btn_stop.setEnabled(True)
    train_tab.manually_stopped = False
    train_tab._on_progress(110, 3.3882, 2.8464, 0.21)
    assert train_tab.progress_bar.value() == 110
    assert 'Step: 110' in train_tab.lbl_metrics.text()
    train_tab._stop_training()
    assert train_tab.manually_stopped is True
    assert train_tab.btn_stop.isEnabled() is False
    assert train_tab.lbl_status.text() == 'Engine State: Idle'
    assert train_tab.progress_bar.value() == 0
    assert train_tab.lbl_metrics.text() == 'Step: 0 | Total Loss: - | Policy Loss: - | Top-1 Accuracy: -'
    train_tab._on_progress(111, 3.35, 2.8, 0.22)
    assert train_tab.progress_bar.value() == 0
    assert train_tab.lbl_metrics.text() == 'Step: 0 | Total Loss: - | Policy Loss: - | Top-1 Accuracy: -'
    train_tab._on_finished()
    assert train_tab.btn_train.isEnabled() is True
    assert train_tab.btn_stop.isEnabled() is False
    assert train_tab.lbl_status.text() == 'Engine State: Idle'
    assert train_tab.progress_bar.value() == 0
    assert train_tab.lbl_metrics.text() == 'Step: 0 | Total Loss: - | Policy Loss: - | Top-1 Accuracy: -'

def test_dataset_tab_stop_resets_state_and_counters(qapp):
    win = MainWindow()
    ds_tab = win.dataset_tab
    ds_tab.lbl_status.setText('Generator State: Generating Stockfish Self-Play Dataset')
    ds_tab.progress_bar.setRange(0, 100)
    ds_tab.btn_generate.setEnabled(False)
    ds_tab.btn_stop.setEnabled(True)
    ds_tab.manually_stopped = False
    ds_tab._on_gen_progress(10, 100, 'Generating game 10/100')
    assert ds_tab.progress_bar.value() == 10
    assert '10/100' in ds_tab.lbl_metrics.text()
    ds_tab._stop_generation()
    assert ds_tab.manually_stopped is True
    assert ds_tab.btn_stop.isEnabled() is False
    assert ds_tab.lbl_status.text() == 'Generator State: Idle'
    assert ds_tab.progress_bar.value() == 0
    assert 'Idle' in ds_tab.lbl_metrics.text()
    ds_tab._on_gen_finished(10)
    assert ds_tab.btn_generate.isEnabled() is True
    assert ds_tab.btn_stop.isEnabled() is False
    assert ds_tab.lbl_status.text() == 'Generator State: Idle'
    assert ds_tab.progress_bar.value() == 0
    assert 'Idle' in ds_tab.lbl_metrics.text()

def test_match_tab_stop(qapp):
    win = MainWindow()
    match_tab = win.match_tab
    match_tab.btn_start.setEnabled(False)
    match_tab.btn_stop.setEnabled(True)
    match_tab.lbl_game_status.setText('Match running...')
    match_tab._stop_match()
    assert match_tab.manually_stopped is True
    assert match_tab.btn_stop.isEnabled() is False
    assert 'Stopped' in match_tab.lbl_game_status.text()
    match_tab._on_match_finished()
    assert match_tab.btn_start.isEnabled() is True
    assert match_tab.btn_stop.isEnabled() is False
    assert match_tab.lbl_game_status.text() == 'Idle'

def test_stockfish_selection_and_sync(qapp):
    win = MainWindow()
    assert hasattr(win.match_tab, 'edit_stockfish_path')
    assert hasattr(win.match_tab, 'btn_browse_stockfish')
    assert hasattr(win.dataset_tab, 'edit_stockfish_path')
    assert hasattr(win.dataset_tab, 'btn_browse_stockfish')
    test_path = 'C:/engines/stockfish_v16.exe'
    win.dataset_tab.edit_stockfish_path.setText(test_path)
    assert win.stockfish_path == test_path
    assert win.match_tab.edit_stockfish_path.text() == test_path
    assert win.play_tab.stockfish_path == test_path
    alt_path = 'C:/engines/stockfish_custom.exe'
    win.match_tab.edit_stockfish_path.setText(alt_path)
    assert win.stockfish_path == alt_path
    assert win.dataset_tab.edit_stockfish_path.text() == alt_path
    assert win.play_tab.stockfish_path == alt_path
