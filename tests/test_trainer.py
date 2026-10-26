import os
import torch
import torch.nn.functional as F
from chess_ai.model import ChessNet, save_checkpoint, load_model
from chess_ai.trainer import WSDScheduler, CosineScheduler, OneCycleScheduler, get_lr_scheduler, build_optimizer, EMAHelper, Trainer

def test_wsd_scheduler_stages():
    model = ChessNet(num_blocks=2, num_filters=32)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    sched = WSDScheduler(opt, base_lr=0.001, min_lr=1e-05, total_steps=100, warmup_ratio=0.1, decay_ratio=0.2)
    lr_1 = sched.step(1)
    assert lr_1 < 0.001
    assert lr_1 > 1e-05
    lr_50 = sched.step(50)
    assert abs(lr_50 - 0.001) < 1e-07
    lr_95 = sched.step(95)
    assert lr_95 < 0.001
    assert lr_95 >= 1e-05

def test_optimizer_parameter_groups():
    model = ChessNet(num_blocks=2, num_filters=32)
    opt = build_optimizer(model, lr=0.001, weight_decay=0.0001)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]['weight_decay'] == 0.0001
    assert opt.param_groups[1]['weight_decay'] == 0.0
    assert len(opt.param_groups[0]['params']) > 0
    assert len(opt.param_groups[1]['params']) > 0

def test_ema_update_and_apply():
    model = ChessNet(num_blocks=2, num_filters=32)
    ema = EMAHelper(model, decay=0.99)
    initial_p = next(iter(ema.shadow.values())).clone()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model, step=100)
    updated_p = next(iter(ema.shadow.values()))
    assert not torch.allclose(initial_p, updated_p)
    target_model = ChessNet(num_blocks=2, num_filters=32)
    ema.apply_to(target_model)
    target_p = next(iter(target_model.parameters()))
    assert torch.allclose(target_p.cpu(), updated_p.cpu())

def test_ema_gpu_update_and_state_dict():
    if not torch.cuda.is_available():
        return
    model = ChessNet(num_blocks=2, num_filters=32).cuda()
    ema = EMAHelper(model, decay=0.99)
    for p in ema.shadow.values():
        assert p.is_cuda
    with torch.no_grad():
        for p in model.parameters():
            p.add_(2.0)
    ema.update(model, step=50)
    sd = ema.state_dict()
    for p in sd.values():
        assert not p.is_cuda

def test_build_optimizer_fused():
    model = ChessNet(num_blocks=2, num_filters=32)
    opt_cpu = build_optimizer(model)
    assert opt_cpu is not None
    if torch.cuda.is_available():
        model_cuda = ChessNet(num_blocks=2, num_filters=32).cuda()
        opt_cuda = build_optimizer(model_cuda)
        assert opt_cuda is not None

def test_checkpoint_v2_roundtrip(tmp_path):
    ckpt_path = str(tmp_path / 'best_model.pt')
    model = ChessNet(num_blocks=4, num_filters=64)
    ema = EMAHelper(model)
    save_checkpoint(ckpt_path, model, config=model.get_config(), ema_state=ema.state_dict(), step=42, metrics={'top1': 0.45})
    loaded_model, meta = load_model(ckpt_path, device='cpu', prefer_ema=True)
    assert meta['format_version'] == 2
    assert meta['model_config']['num_blocks'] == 4
    assert meta['model_config']['num_filters'] == 64
    assert meta['step'] == 42
    assert meta['metrics']['top1'] == 0.45
    assert loaded_model.num_blocks == 4
    assert loaded_model.num_filters == 64

def test_trainer_resume_restores_model_weights(tmp_path):
    from chess_ai.dataset_generator import save_records_to_parquet
    parquet_path = str(tmp_path / 'dummy.parquet')
    records = [{'date': '2026.10.23', 'white_elo': 2500, 'black_elo': 2500, 'winner': 'white', 'moves_uci': ['e2e4', 'e7e5', 'g1f3', 'b8c6'] * 5, 'end_type': 'checkmate', 'source': 'test'}] * 50
    save_records_to_parquet(records, parquet_path, append=False)
    ckpt_path = str(tmp_path / 'saved_ckpt.pt')
    model_orig = ChessNet(num_blocks=2, num_filters=32)
    with torch.no_grad():
        for p in model_orig.parameters():
            p.fill_(0.42)
    save_checkpoint(ckpt_path, model_orig, config=model_orig.get_config(), step=10)
    model_new = ChessNet(num_blocks=2, num_filters=32)
    with torch.no_grad():
        for p in model_new.parameters():
            p.zero_()
    trainer = Trainer(model=model_new, parquet_path=parquet_path, save_dir=str(tmp_path / 'ckpts'), batch_size=8, num_workers=0, device='cpu')
    trainer.train(max_steps=11, resume_checkpoint=ckpt_path)
    trainer.close()
    p0 = next(model_new.parameters())
    assert not torch.allclose(p0, torch.zeros_like(p0))

def test_trainer_overfit_batch():
    torch.manual_seed(42)
    model = ChessNet(num_blocks=2, num_filters=32)
    opt = build_optimizer(model, lr=0.005)
    states = torch.randn(8, 112, 8, 8)
    target_p = torch.randint(0, 4096, (8,))
    target_v = F.softmax(torch.randn(8, 3), dim=-1)
    legal_masks = torch.zeros(8, 4096, dtype=torch.bool)
    for i in range(8):
        legal_masks[i, target_p[i]] = True
        legal_masks[i, (target_p[i] + 1) % 4096] = True
    initial_loss = None
    final_loss = None
    for step in range(30):
        opt.zero_grad()
        p_logits, v_logits = model(states)
        masked_p = p_logits.float().view(-1, 4096).masked_fill(~legal_masks, -10000.0)
        loss_p = F.cross_entropy(masked_p, target_p)
        loss_v = F.cross_entropy(v_logits.float(), target_v)
        loss = loss_p + 0.5 * loss_v
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        opt.step()
        final_loss = loss.item()
    assert final_loss < initial_loss * 0.4, f'Loss did not drop enough: {initial_loss} -> {final_loss}'

def test_cosine_and_onecycle_schedulers():
    model = ChessNet(num_blocks=2, num_filters=32)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    cos_sched = get_lr_scheduler('cosine', opt, base_lr=0.001, total_steps=100)
    assert isinstance(cos_sched, CosineScheduler)
    lr_start = cos_sched.step(1)
    lr_mid = cos_sched.step(50)
    lr_end = cos_sched.step(100)
    assert lr_start < lr_mid
    assert lr_end < lr_mid
    oc_sched = get_lr_scheduler('onecycle', opt, base_lr=0.001, total_steps=100)
    assert isinstance(oc_sched, OneCycleScheduler)
    lr_oc_start = oc_sched.step(1)
    lr_oc_peak = oc_sched.step(30)
    lr_oc_end = oc_sched.step(100)
    assert lr_oc_start < lr_oc_peak
    assert lr_oc_end < lr_oc_peak
    assert abs(lr_oc_peak - 0.001) < 0.0001

def test_trainer_init_resume_args(tmp_path):
    from chess_ai.dataset_generator import save_records_to_parquet
    parquet_path = str(tmp_path / 'dummy.parquet')
    save_records_to_parquet([{'date': '2026.10.23', 'white_elo': 2500, 'black_elo': 2500, 'winner': 'white', 'moves_uci': ['e2e4', 'e7e5'], 'end_type': 'checkmate', 'source': 'test'}], parquet_path, append=False)
    save_dir = str(tmp_path / 'ckpts')
    model = ChessNet(num_blocks=2, num_filters=32)
    t1 = Trainer(model=model, parquet_path=parquet_path, save_dir=save_dir, resume=False)
    assert t1.resume_checkpoint is None
    t1.close()
    t2 = Trainer(model=model, parquet_path=parquet_path, save_dir=save_dir, resume=True)
    assert t2.resume_checkpoint == os.path.join(save_dir, 'last.pt')
    t2.close()
    custom_path = str(tmp_path / 'custom.pt')
    t3 = Trainer(model=model, parquet_path=parquet_path, save_dir=save_dir, resume=custom_path)
    assert t3.resume_checkpoint == custom_path
    t3.close()
    t4 = Trainer(model=model, parquet_path=parquet_path, save_dir=save_dir, resume_checkpoint=custom_path)
    assert t4.resume_checkpoint == custom_path
    t4.close()

def test_trainer_resume_from_init(tmp_path):
    from chess_ai.dataset_generator import save_records_to_parquet
    parquet_path = str(tmp_path / 'dummy.parquet')
    records = [{'date': '2026.10.23', 'white_elo': 2500, 'black_elo': 2500, 'winner': 'white', 'moves_uci': ['e2e4', 'e7e5', 'g1f3', 'b8c6'] * 5, 'end_type': 'checkmate', 'source': 'test'}] * 20
    save_records_to_parquet(records, parquet_path, append=False)
    ckpt_path = str(tmp_path / 'init_ckpt.pt')
    model_orig = ChessNet(num_blocks=2, num_filters=32)
    with torch.no_grad():
        for p in model_orig.parameters():
            p.fill_(0.77)
    save_checkpoint(ckpt_path, model_orig, config=model_orig.get_config(), step=5)
    model_new = ChessNet(num_blocks=2, num_filters=32)
    with torch.no_grad():
        for p in model_new.parameters():
            p.zero_()
    trainer = Trainer(model=model_new, parquet_path=parquet_path, save_dir=str(tmp_path / 'ckpts_init'), batch_size=8, num_workers=0, device='cpu', resume=ckpt_path)
    trainer.train(max_steps=6)
    trainer.close()
    p0 = next(model_new.parameters())
    assert not torch.allclose(p0, torch.zeros_like(p0))

def test_trainer_resume_nonexistent(tmp_path):
    from chess_ai.dataset_generator import save_records_to_parquet
    parquet_path = str(tmp_path / 'dummy.parquet')
    records = [{'date': '2026.10.23', 'white_elo': 2500, 'black_elo': 2500, 'winner': 'white', 'moves_uci': ['e2e4', 'e7e5'], 'end_type': 'checkmate', 'source': 'test'}] * 10
    save_records_to_parquet(records, parquet_path, append=False)
    logs = []
    model = ChessNet(num_blocks=2, num_filters=32)
    trainer = Trainer(model=model, parquet_path=parquet_path, save_dir=str(tmp_path / 'ckpts_nonexistent'), batch_size=8, num_workers=0, device='cpu', resume=str(tmp_path / 'missing_file.pt'))
    trainer.train(max_steps=2, on_log=lambda msg: logs.append(msg))
    trainer.close()
    assert any(('not found' in msg for msg in logs))

def test_trainer_resume_when_checkpoint_step_exceeds_max_steps(tmp_path):
    from chess_ai.dataset_generator import save_records_to_parquet
    parquet_path = str(tmp_path / 'dummy.parquet')
    records = [{'date': '2026.10.23', 'white_elo': 2500, 'black_elo': 2500, 'winner': 'white', 'moves_uci': ['e2e4', 'e7e5'], 'end_type': 'checkmate', 'source': 'test'}] * 10
    save_records_to_parquet(records, parquet_path, append=False)
    ckpt_path = str(tmp_path / 'init_ckpt.pt')
    model_orig = ChessNet(num_blocks=2, num_filters=32)
    save_checkpoint(ckpt_path, model_orig, config=model_orig.get_config(), step=180020)
    save_dir = str(tmp_path / 'ckpts_resume_large')
    trainer = Trainer(model=ChessNet(num_blocks=2, num_filters=32), parquet_path=parquet_path, save_dir=save_dir, batch_size=8, num_workers=0, device='cpu')
    logs = []
    trainer.train(max_steps=2, resume_checkpoint=ckpt_path, on_log=lambda msg: logs.append(msg))
    trainer.close()
    last_pt = os.path.join(save_dir, 'last.pt')
    assert os.path.exists(last_pt)
    data = torch.load(last_pt, map_location='cpu')
    assert data['step'] == 180022, f"Expected step 180022 but got {data['step']}"

def test_resume_corrupt_checkpoint_fallback(tmp_path):
    from chess_ai.dataset_generator import save_records_to_parquet
    parquet_path = str(tmp_path / 'dummy.parquet')
    records = [{'date': '2026.10.23', 'white_elo': 2500, 'black_elo': 2500, 'winner': 'white', 'moves_uci': ['e2e4', 'e7e5'], 'end_type': 'checkmate', 'source': 'test'}] * 10
    save_records_to_parquet(records, parquet_path, append=False)
    save_dir = str(tmp_path / 'ckpts_fallback')
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, 'best_model.pt')
    m_clean = ChessNet(num_blocks=2, num_filters=32)
    save_checkpoint(best_path, m_clean, config=m_clean.get_config(), step=180800)
    last_path = os.path.join(save_dir, 'last.pt')
    m_corrupt = ChessNet(num_blocks=2, num_filters=32)
    with torch.no_grad():
        for p in m_corrupt.parameters():
            p.fill_(float('nan'))
    save_checkpoint(last_path, m_corrupt, config=m_corrupt.get_config(), step=180884)
    logs = []
    trainer = Trainer(model=ChessNet(num_blocks=2, num_filters=32), parquet_path=parquet_path, save_dir=save_dir, batch_size=8, num_workers=0, device='cpu')
    trainer.train(steps=1, resume_checkpoint=last_path, on_log=lambda msg: logs.append(msg))
    trainer.close()
    assert any(('Falling back to' in log for log in logs))
    last_saved = torch.load(last_path, map_location='cpu')
    for v in last_saved['model_state'].values():
        assert not torch.isnan(v).any()

def test_rmsnorm_numerical_stability():
    from chess_ai.model import RMSNorm
    norm = RMSNorm(dim=64, eps=1e-06)
    x = torch.zeros(4, 64, 8, 8, requires_grad=True)
    out = norm(x)
    loss = out.sum()
    loss.backward()
    assert torch.isfinite(x.grad).all()

def test_trainer_resume_explicit_steps_argument(tmp_path):
    from chess_ai.dataset_generator import save_records_to_parquet
    parquet_path = str(tmp_path / 'dummy.parquet')
    records = [{'date': '2026.10.23', 'white_elo': 2500, 'black_elo': 2500, 'winner': 'white', 'moves_uci': ['e2e4', 'e7e5'], 'end_type': 'checkmate', 'source': 'test'}] * 10
    save_records_to_parquet(records, parquet_path, append=False)
    ckpt_path = str(tmp_path / 'init_ckpt.pt')
    model_orig = ChessNet(num_blocks=2, num_filters=32)
    save_checkpoint(ckpt_path, model_orig, config=model_orig.get_config(), step=180020)
    save_dir = str(tmp_path / 'ckpts_explicit_steps')
    trainer = Trainer(model=ChessNet(num_blocks=2, num_filters=32), parquet_path=parquet_path, save_dir=save_dir, batch_size=8, num_workers=0, device='cpu')
    trainer.train(steps=2, resume_checkpoint=ckpt_path)
    trainer.close()
    last_pt = os.path.join(save_dir, 'last.pt')
    assert os.path.exists(last_pt)
    data = torch.load(last_pt, map_location='cpu')
    assert data['step'] == 180022, f"Expected step 180022 but got {data['step']}"

def test_resume_corrupt_checkpoint_fallback(tmp_path):
    from chess_ai.dataset_generator import save_records_to_parquet
    parquet_path = str(tmp_path / 'dummy.parquet')
    records = [{'date': '2026.10.23', 'white_elo': 2500, 'black_elo': 2500, 'winner': 'white', 'moves_uci': ['e2e4', 'e7e5'], 'end_type': 'checkmate', 'source': 'test'}] * 10
    save_records_to_parquet(records, parquet_path, append=False)
    save_dir = str(tmp_path / 'ckpts_fallback')
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, 'best_model.pt')
    m_clean = ChessNet(num_blocks=2, num_filters=32)
    save_checkpoint(best_path, m_clean, config=m_clean.get_config(), step=180800)
    last_path = os.path.join(save_dir, 'last.pt')
    m_corrupt = ChessNet(num_blocks=2, num_filters=32)
    with torch.no_grad():
        for p in m_corrupt.parameters():
            p.fill_(float('nan'))
    save_checkpoint(last_path, m_corrupt, config=m_corrupt.get_config(), step=180884)
    logs = []
    trainer = Trainer(model=ChessNet(num_blocks=2, num_filters=32), parquet_path=parquet_path, save_dir=save_dir, batch_size=8, num_workers=0, device='cpu')
    trainer.train(steps=1, resume_checkpoint=last_path, on_log=lambda msg: logs.append(msg))
    trainer.close()
    assert any(('Falling back to' in log for log in logs))
    last_saved = torch.load(last_path, map_location='cpu')
    for v in last_saved['model_state'].values():
        assert not torch.isnan(v).any()

def test_rmsnorm_numerical_stability():
    from chess_ai.model import RMSNorm
    norm = RMSNorm(dim=64, eps=1e-06)
    x = torch.zeros(4, 64, 8, 8, requires_grad=True)
    out = norm(x)
    loss = out.sum()
    loss.backward()
    assert torch.isfinite(x.grad).all()
