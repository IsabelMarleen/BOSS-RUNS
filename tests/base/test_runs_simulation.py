import logging
import subprocess
import time
from pathlib import Path

import boss.config
import boss.runs.simulation
import numpy as np
import pytest

from ..constants import PATHS

barcode_list = [[None, 100], [None, 1000], [["barcode01", "barcode02"], 100], [["barcode01", "barcode02"], 1000]]

@pytest.fixture(params=barcode_list, ids=["not barcoded, small window", "not barcoded, large window", "barcoded, small window", "barcoded, large window"])
def args(request):
    conf = boss.config.Config()
    args = conf.args
    # assign some args since we don't load the full config
    args.general.ref = PATHS.fasta
    args.simulation.fq = PATHS.fastq
    args.simulation.paf_full = PATHS.paf
    args.simulation.paf_trunc = PATHS.paf_trunc
    args.simulation.maxb = 8
    args.simulation.batchsize = 100
    args.simulation.dumptime = 10000
    args.general.barcodes = request.param[0]
    args.optional.window_size = request.param[1]
    return args



def test_init(args):
    b = boss.runs.simulation.BossRunsSim(args=args)
    b.init_sim()
    assert type(b.ref) is boss.runs.reference.Reference  # type: ignore
    assert Path(f"{b.ref.ref}.mmi").is_file()
    assert len(b.contigs) == 9
    logging.info(b.contigs.keys())
    assert b.contigs["NZ_CP041015.1"].length == 4045619
    assert Path("00_reads/control_0.fa").is_file()
    assert Path("00_reads/boss_0.fa").is_file()
    subprocess.run('rm -r 00_reads/', shell=True)



def test_process_batch(args):
    args.simulation.batchsize = 500
    args.simulation.maxb = 9
    b = boss.runs.simulation.BossRunsSim(args=args)
    b.init_sim()
    assert b.batch == 0
    tic = time.time()
    # we need to switch bucket switches manually here
    for cname, cont in b.contigs_filt.items():
        cont.switched_on = np.ones(shape=(b.nbarcodes), dtype="bool") 
    next_update = b.process_batch_sim(b.process_batch_runs_sim)
    assert b.batch == 1
    assert next_update != b.args.general.wait
    # check that new strats were produced
    assert Path("out_boss/masks/boss.npz").stat().st_mtime > tic
    assert Path("00_reads/control_1.fa").is_file()
    assert Path("00_reads/boss_1.fa").is_file()
    # run another batch to check rejections
    tic = time.time()
    next_update = b.process_batch_sim(b.process_batch_runs_sim)
    assert b.batch == 2
    assert next_update != b.args.general.wait
    # check that new strats were produced
    assert Path("out_boss/masks/boss.npz").stat().st_mtime > tic
    assert Path("00_reads/control_2.fa").is_file()
    assert Path("00_reads/boss_2.fa").is_file()
    assert b.read_cache.time_boss < b.read_cache.time_control
    subprocess.run('rm -r 00_reads/', shell=True)



