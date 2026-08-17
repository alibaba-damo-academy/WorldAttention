"""Page residency: pool reuse, LRU demotion, NVMe spill and reload."""
import os
import shutil
import sys
import tempfile

import torch

from pipeline.hkv import HKVTierManager

LAYERS, BATCH, PAGE_TOKENS, HEADS, HEAD_DIM = 2, 1, 4, 1, 2
GEOMETRY = dict(
    layers=LAYERS, batch=BATCH, page_tokens=PAGE_TOKENS,
    heads=HEADS, head_dim=HEAD_DIM, dtype=torch.float32,
)


def _manager(device="cpu", **kwargs):
    tier = HKVTierManager(**kwargs)
    tier.configure(device=torch.device(device), **GEOMETRY)
    return tier


def _fill(tier, key, value):
    tier.allocate(key).fill_(float(value))


def _value(page):
    return float(page.flatten()[0].item())


def test_a_page_round_trips_through_the_cpu_tier():
    tier = _manager()
    for page_id in range(3):
        _fill(tier, (0, page_id), page_id + 1)
    for page_id in range(3):
        assert tier.location((0, page_id)) == "cpu"
        assert _value(tier.get((0, page_id))) == page_id + 1
    assert tier.get((9, 9)) is None
    assert tier.location((9, 9)) == "absent"


def test_cpu_slots_are_reused_rather_than_reallocated():
    """The steady state must not allocate: fresh host buffers every block were the old bottleneck."""
    tier = _manager(cpu_max_pages=2)
    for page_id in range(6):
        _fill(tier, (0, page_id), page_id)
    assert tier._cpu_allocated == 2
    assert len(tier._cpu) <= 2


def test_over_capacity_drops_the_least_recently_used_without_nvme():
    tier = _manager(cpu_max_pages=2)
    _fill(tier, (0, 0), 1)
    _fill(tier, (0, 1), 2)
    tier.get((0, 0))
    _fill(tier, (0, 2), 3)

    assert tier.location((0, 1)) == "absent"
    assert tier.location((0, 0)) == "cpu"
    assert tier.stats()["drops"] == 1


def test_spilled_pages_come_back_from_nvme():
    tmp = tempfile.mkdtemp()
    try:
        tier = _manager(cpu_max_pages=1, nvme_enabled=True, nvme_dir=tmp)
        _fill(tier, (0, 0), 7)
        _fill(tier, (0, 1), 8)

        assert tier.location((0, 0)) == "nvme"
        assert _value(tier.get((0, 0))) == 7
        assert tier.stats()["spills"] >= 1 and tier.stats()["loads"] == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_an_unchanged_page_is_written_to_nvme_once():
    tmp = tempfile.mkdtemp()
    try:
        tier = _manager(cpu_max_pages=1, nvme_enabled=True, nvme_dir=tmp)
        _fill(tier, (0, 0), 1)
        for page_id in range(1, 4):
            _fill(tier, (0, page_id), page_id)
            tier.get((0, 0))
        assert tier.stats()["spills"] >= 3
        assert len(os.listdir(tmp)) == 4
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cleanup_removes_the_spilled_files():
    tmp = tempfile.mkdtemp()
    try:
        tier = _manager(cpu_max_pages=1, nvme_enabled=True, nvme_dir=tmp)
        _fill(tier, (0, 0), 1)
        _fill(tier, (0, 1), 2)
        assert os.listdir(tmp)
        tier.cleanup()
        assert os.listdir(tmp) == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_clear_keeps_the_pools_but_forgets_the_pages():
    tier = _manager(cpu_max_pages=4)
    for page_id in range(3):
        _fill(tier, (0, page_id), page_id)
    allocated = tier._cpu_allocated
    tier.clear()

    assert tier.location((0, 0)) == "absent"
    assert tier._cpu_allocated == allocated
    _fill(tier, (1, 0), 5)
    assert tier._cpu_allocated == allocated


def test_discard_returns_the_storage():
    tier = _manager(cpu_max_pages=4)
    _fill(tier, (0, 0), 1)
    tier.discard((0, 0))
    assert tier.location((0, 0)) == "absent"
    _fill(tier, (0, 1), 2)
    assert tier._cpu_allocated == 1


def test_gpu_tier_takes_new_pages_and_demotes_the_oldest():
    if not torch.cuda.is_available():
        print("SKIP (no CUDA)", end=" ")
        return
    tier = _manager(device="cuda", gpu_max_pages=2, cpu_max_pages=4)
    _fill(tier, (0, 0), 1)
    _fill(tier, (0, 1), 2)
    assert tier.location((0, 0)) == "gpu" and tier.location((0, 1)) == "gpu"

    _fill(tier, (0, 2), 3)
    assert tier.location((0, 0)) == "cpu"
    assert tier.location((0, 2)) == "gpu"
    assert tier.stats()["demotions"] == 1
    assert _value(tier.get((0, 0))) == 1


def test_gpu_tier_does_not_evict_to_serve_a_read():
    """Promoting on a read must never push out a page the same install still needs."""
    if not torch.cuda.is_available():
        print("SKIP (no CUDA)", end=" ")
        return
    tier = _manager(device="cuda", gpu_max_pages=1, cpu_max_pages=4)
    _fill(tier, (0, 0), 1)
    _fill(tier, (0, 1), 2)
    assert tier.location((0, 1)) == "gpu"

    assert _value(tier.get((0, 0))) == 1
    assert tier.location((0, 0)) == "cpu"
    assert tier.location((0, 1)) == "gpu"
    assert tier.stats()["promotions"] == 0

if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(dict(globals()).items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print("PASS", name)
        except Exception as err:
            failures += 1
            print("FAIL", name, "->", err)
    print("\nRESULT:", "ALL PASS" if failures == 0 else f"{failures} FAILURE(S)")
    sys.exit(1 if failures else 0)
