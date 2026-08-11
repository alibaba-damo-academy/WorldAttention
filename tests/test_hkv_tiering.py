"""Multi-tier residency: LRU order, NVMe spill and reload, backups. CPU only."""
import os
import shutil
import sys
import tempfile

import torch

from pipeline.hkv import HKVTierManager


def _blob(tag: int) -> dict:
    return {"k": [torch.full((2, 2), float(tag))], "v": [torch.full((2, 2), -float(tag))]}


def test_resident_chunks_round_trip():
    tier = HKVTierManager(cpu_max_chunks=4)
    for cid in range(3):
        tier.put(cid, _blob(cid))
    for cid in range(3):
        assert tier.location(cid) == "cpu"
        assert torch.equal(tier.get(cid)["k"][0], torch.full((2, 2), float(cid)))
    assert tier.get(99) is None
    assert tier.location(99) == "absent"


def test_over_capacity_drops_the_least_recently_used_without_nvme():
    tier = HKVTierManager(cpu_max_chunks=2)
    tier.put(0, _blob(0))
    tier.put(1, _blob(1))
    tier.get(0)                      # 0 becomes the most recently used
    tier.put(2, _blob(2))

    assert tier.location(1) == "absent"
    assert tier.location(0) == "cpu"
    assert tier.location(2) == "cpu"
    assert tier.stats()["n_drops"] == 1
    assert tier.stats()["n_spills"] == 0


def test_capacity_is_never_below_one():
    tier = HKVTierManager(cpu_max_chunks=0)
    tier.put(0, _blob(0))
    assert tier.location(0) == "cpu"


def test_nvme_spills_and_reloads():
    tmp = tempfile.mkdtemp()
    try:
        tier = HKVTierManager(cpu_max_chunks=2, nvme_enabled=True, nvme_dir=tmp)
        for cid in range(3):
            tier.put(cid, _blob(cid))

        assert tier.location(0) == "nvme"
        assert tier.stats()["n_spills"] == 1
        assert len(os.listdir(tmp)) == 1

        reloaded = tier.get(0)
        assert torch.equal(reloaded["k"][0], torch.full((2, 2), 0.0))
        assert tier.location(0) == "cpu"
        assert tier.stats()["n_loads"] == 1
        assert tier.stats()["n_drops"] == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_nvme_keeps_a_full_backup_so_respilling_is_free():
    tmp = tempfile.mkdtemp()
    try:
        tier = HKVTierManager(cpu_max_chunks=1, nvme_enabled=True, nvme_dir=tmp)
        tier.put(0, _blob(0))
        tier.put(1, _blob(1))            # spills 0, writes its backup
        assert tier.stats()["n_spills"] == 1

        tier.get(0)                      # reload 0, spilling 1
        tier.put(2, _blob(2))            # spills 0 again, reusing the existing backup

        assert tier.stats()["backups"] == len(os.listdir(tmp))
        assert torch.equal(tier.get(0)["k"][0], torch.full((2, 2), 0.0))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_nvme_requires_a_directory():
    try:
        HKVTierManager(cpu_max_chunks=1, nvme_enabled=True)
    except ValueError:
        pass
    else:
        raise AssertionError("expected a missing nvme_dir to be rejected")


def test_cleanup_removes_files_and_state():
    tmp = tempfile.mkdtemp()
    try:
        tier = HKVTierManager(cpu_max_chunks=1, nvme_enabled=True, nvme_dir=tmp)
        tier.put(0, _blob(0))
        tier.put(1, _blob(1))
        assert len(os.listdir(tmp)) == 1

        tier.cleanup()
        assert os.listdir(tmp) == []
        assert tier.stats()["cpu_resident"] == 0
        assert tier.location(0) == "absent"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(dict(globals()).items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print("PASS", name)
        except Exception as err:  # noqa: BLE001
            failures += 1
            print("FAIL", name, "->", err)
    print("\nRESULT:", "ALL PASS" if failures == 0 else f"{failures} FAILURE(S)")
    sys.exit(1 if failures else 0)
