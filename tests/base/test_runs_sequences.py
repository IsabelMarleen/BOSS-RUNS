import numpy as np
import pytest

import boss.runs.sequences as brs


# with deletions (default) there are 5 states
# which means 15 genotypes for diploids
@pytest.mark.parametrize('ploidy, b, g', [
    (1, 4, 5),
    (2, 4, 15)
])
def test_priors(ploidy, b, g):
    p = brs.Priors(ploidy=ploidy)
    assert p.len_b == b + 1
    assert p.len_g == g
    assert p.phi_stored.shape == (b + 1, g, 1000)
    assert p.priors.shape == (b, g)


@pytest.mark.xfail(raises=ValueError)
def test_priors_ploidy():
    brs.Priors(ploidy=3)


def test_uniform():
    p = brs.Priors(ploidy=1)
    p.uniform_priors()
    assert np.all(np.isclose(p.priors, p.priors[0]))


# without deletions there are 4 states
# and 10 genotypes for diploids
@pytest.mark.parametrize('diploid, del_err, b, g', [
    (False, 0, 4, 4),
    (True, 0, 4, 10)
])
def test_generate_phi(diploid, del_err, b, g):
    lenb, leng, phi = brs.Priors._generate_phi(diploid=diploid, deletion_error=del_err)
    assert lenb == b
    assert leng == g
    assert phi.shape == (b, g)



@pytest.mark.parametrize('del_err, b, g', [
    (0, 4, 4),
    (0.5, 4, 5)
])
def test_haploid_priors(del_err, b, g):
    priors = brs.Priors._haploid_priors(deletion_error=del_err)
    assert priors.shape == (b, g)


@pytest.mark.parametrize('del_err, b, g', [
    (0, 4, 10),
    (0.5, 4, 15)
])
def test_diploid_priors(del_err, b, g):
    priors = brs.Priors._diploid_priors(deletion_error=del_err)
    assert priors.shape == (b, g)



def test_convert_records(zymo_read_batch, paf_dict, zymo_ref):
    cc = brs.CoverageConverter()

    incr = cc.convert_records(
        paf_dict=paf_dict,
        seqs=zymo_read_batch.read_sequences,
        quals=zymo_read_batch.read_qualities
    )

    assert incr
    assert len(incr) == 8
    # check that correct ref is present
    assert incr['NZ_CP041013.1']
    # correct number of reads from that ref
    assert len(incr['NZ_CP041013.1']) == 469
    # coordinates on the ref of the first mapping
    assert incr['NZ_CP041013.1'][0][: 2] == (4245280, 4254995)
    # seq of the first mapping on the ref
    assert np.allclose(incr['NZ_CP041013.1'][0][2][:20],
                       np.array([0, 1, 1, 2, 1, 2, 1, 0, 1, 1, 2, 0, 1, 2, 0, 0, 0, 0, 1, 1],
                             dtype=np.uint8))
    # qual threshold for that sequence
    assert np.sum(incr['NZ_CP041013.1'][0][3][:20]) == 20
    # collect the ration of positions that are different from the reference
    nonref_fraction = []
    for refname, adds in incr.items():
        for i in range(len(adds)):
            start, end, seq, _qual, _ = adds[i]
            # grab the reference sequence
            refarr = zymo_ref.contigs[refname].seq_int[start: end]
            # check where the mapping differs
            nonref = np.where(refarr != seq)[0].shape[0]
            nonref_fraction.append(nonref / len(refarr))
    nonref_fraction = np.array(nonref_fraction)
    # how many of the mappings have more than 15 percent divergence
    lessthan_p15_nonref = np.where(nonref_fraction < 0.15)[0].shape[0]
    # the total fraction of reads with <15% div should be that number
    assert np.allclose(lessthan_p15_nonref / len(nonref_fraction), 0.859758771929)





@pytest.fixture
def scoring():
    return brs.Scoring()


def test_init_scoring(scoring):
    assert np.isclose(scoring.score0, 0.04969294)
    assert np.isclose(scoring.ent0, 0.09302521)


def test_score_array(scoring):
    scoring.init_score_array()
    assert scoring.score_arr.shape == (40, 40, 40, 40, 40, 4)
    assert scoring.entropy_arr.shape == (40, 40, 40, 40, 40, 4)
    assert np.isclose(scoring.score_arr[28, 0, 0, 0, 0, 3], 3.834200141940696e-44)
    assert np.isclose(scoring.entropy_arr[28, 0, 0, 0, 0, 3], 3.834200141940696e-44)
    assert np.isclose(scoring.score_arr[2, 0, 0, 0, 0, 3], 0.17253973305650225)
    assert np.isclose(scoring.entropy_arr[2, 0, 0, 0, 0, 3], 0.22957118271635163)

@pytest.fixture
def posteriors():
    expected_pos = np.array([[[8.06983293e-01, 2.72813825e-03, 8.03958698e-09, 1.90287643e-01,
         9.18097027e-07],
        [8.09191562e-01, 1.15578307e-10, 8.06158690e-09, 1.90808355e-01,
         7.44852526e-08],
        [8.09191597e-01, 1.15578312e-10, 1.15578312e-10, 1.90808363e-01,
         3.97254697e-08]],
       [[2.72813825e-03, 8.06983293e-01, 8.03958698e-09, 1.90287643e-01,
         9.18097027e-07],
        [1.41342671e-02, 1.76642261e-07, 4.16524602e-08, 9.85865130e-01,
         3.84849045e-07],
        [1.41342702e-02, 1.76642300e-07, 5.97168020e-10, 9.85865347e-01,
         2.05252869e-07]],

       [[1.39370475e-02, 1.39370475e-02, 1.21488820e-05, 9.72109066e-01,
         4.69021756e-06],
        [1.41340960e-02, 5.97160661e-10, 1.23206486e-05, 9.85853198e-01,
         3.84844387e-07],
        [1.41342702e-02, 5.97168020e-10, 1.76642300e-07, 9.85865347e-01,
         2.05252869e-07]],

       [[4.84635825e-05, 4.84635825e-05, 1.42817977e-10, 9.99903056e-01,
         1.63093901e-08],
        [4.84659320e-05, 2.04766884e-12, 1.42824901e-10, 9.99951533e-01,
         1.31963458e-09],
        [4.84659321e-05, 2.04766884e-12, 2.04766884e-12, 9.99951533e-01,
         7.03805110e-10]]])
    return expected_pos

def test_calc_posteriors(scoring, posteriors):
    pos = scoring.calc_posterior(np.array([[4, 4, 1, 5, 0],[4, 0, 1, 5, 0],[4, 0, 0, 5, 0]])) # 4 bases, N
    
    assert pos.shape == posteriors.shape == (4,3,5)
    np.testing.assert_allclose(pos, posteriors)


def test_calc_scores(scoring, posteriors):
    # Calculate 3 new scores
    n = 3
    score0 = 0.04969294
    init_scores = np.repeat(score0, repeats=n, axis=0)
    scores, entropy = scoring.calc_score(init_scores, posteriors[0,:,:])

    exp_scores = np.array([0.41063798, 0.40142846, 0.40142834])
    exp_entropy = np.array([0.50490841, 0.48739441, 0.48739369])

    assert scores.shape == exp_scores.shape == (3,)
    assert entropy.shape == exp_entropy.shape == (3,)
    np.testing.assert_allclose(scores, exp_scores)
    np.testing.assert_allclose(entropy, exp_entropy)




