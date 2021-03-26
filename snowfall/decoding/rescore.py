# Copyright (c)  2021  Xiaomi Corp.       (author: Fangjun Kuang)

import logging
import k2
import k2.fsa_properties as fsa_properties
import torch

from snowfall.common import invert_permutation
from snowfall.training.compute_embeddings import compute_embeddings_from_phone_seqs
from snowfall.training.compute_embeddings import create_phone_fsas
from snowfall.training.compute_embeddings import generate_nbest_list_phone_seqs
from snowfall.decoding.util import get_log_probs


def get_paths(lats: k2.Fsa, num_paths: int,
              use_double_scores: bool = True) -> k2.RaggedInt:
    '''Return a n-best list **sampled** from the given lattice.

    Args:
      lats:
        An FsaVec, e.g., the decoding output from the 1st pass.
      num_paths:
        It is the `n` in `n-best`.
      use_double_scores:
        True to use double precision in :func:`k2.random_paths`;
        False to use single precision.
    Returns:
      A ragged tensor with 3 axes: [seq][path][arc_pos] .
    '''
    assert len(lats.shape) == 3

    # paths will be k2.RaggedInt with 3 axes: [seq][path][arc_pos],
    # containing arc_idx012
    paths = k2.random_paths(lats,
                            use_double_scores=use_double_scores,
                            num_paths=num_paths)
    return paths


def get_word_fsas(lats: k2.Fsa, paths: k2.RaggedInt) -> k2.Fsa:
    '''
    Args:
      lats:
        An FsaVec, e.g., from the 1st decoding
      paths:
        Return value of :func:`get_paths`
    '''
    assert len(lats.shape) == 3
    assert hasattr(lats, 'aux_labels')

    # word_seqs will be k2.RaggedInt like paths, but containing words
    # (and final -1's, and 0's for epsilon)
    word_seqs = k2.index(lats.aux_labels, paths)

    # Remove epsilons and -1 from `word_seqs`
    word_seqs = k2.ragged.remove_values_leq(word_seqs, 0)

    seq_to_path_shape = k2.ragged.get_layer(word_seqs.shape(), 0)
    path_to_seq_map = seq_to_path_shape.row_ids(1)

    word_seqs = k2.ragged.remove_axis(word_seqs, 0)

    word_fsas = k2.linear_fsa(word_seqs)

    word_fsas_with_epsilons = k2.add_epsilon_self_loops(word_fsas)
    return word_fsas_with_epsilons, seq_to_path_shape


@torch.no_grad()
def rescore(lats: k2.Fsa,
            paths: k2.RaggedInt,
            word_fsas: k2.Fsa,
            tot_scores_1st: torch.Tensor,
            seq_to_path_shape: k2.RaggedShape,
            ctc_topo: k2.Fsa,
            decoding_graph: k2.Fsa,
            dense_fsa_vec: k2.DenseFsaVec,
            second_pass_model: torch.nn.Module,
            max_phone_id: int,
            use_double_scores: bool = True):
    '''
    Args:
      lats:
        Lattice from the 1st pass decoding with indexes [seq][state][arc].
      paths:
        An FsaVec returned by :func:`get_paths`.
      word_fsas:
        An FsaVec returned by :func:`get_word_fsas`.
      tot_scores_1st:
        Total scores of the paths from the 1st pass.
      ctc_topo:
        The return value of :func:`build_ctc_topo`.
      decoding_graph:
        An Fsa.
      dense_fsa_vec:
        It contains output from the first pass for computing embeddings.
        Note that the output is not processed by log-softmax.
      second_pass_model:
        Model of the second pass.
      use_double_scores:
        True to use double precision in :func:`k2.Fsa.get_tot_scores`;
        False to use single precision.
    Returns:
      Return the best_paths of each seq after rescoring.
    '''
    device = lats.device
    assert hasattr(lats, 'phones')
    assert paths.num_axes() == 3

    # phone_seqs will be k2.RaggedInt like paths, but containing phones
    # (and final -1's, and 0's for epsilon)
    phone_seqs = k2.index(lats.phones, paths)

    # Remove epsilons from `phone_seqs`
    phone_seqs = k2.ragged.remove_values_eq(phone_seqs, 0)

    # padded_embeddings is a 3-D tensor with shape (N, T, C)
    #
    # len_per_path is a 1-D tensor with shape (N,)
    #    len_per_path.shape[0] == N
    #    0 < len_per_path[i] <= T
    #
    # path_to_seq is a 1-D tensor with shape (N,)
    #    path_to_seq.shape[0] == N
    #    0 <= path_to_seq[i] < num_seqs
    #
    # num_repeats is a k2.RaggedInt with two axes [seq][path_multiplicities]
    #
    # CAUTION: Paths within a seq are reordered due to `k2.ragged.unique_sequences`.
    padded_embeddings, len_per_path, path_to_seq, num_repeats, new2old = compute_embeddings_from_phone_seqs(
        phone_seqs=phone_seqs,
        ctc_topo=ctc_topo,
        dense_fsa_vec=dense_fsa_vec,
        max_phone_id=max_phone_id)

    # padded_embeddings is of shape [num_paths, max_phone_seq_len, num_features]
    # i.e., [N, T, C]
    padded_embeddings = padded_embeddings.permute(0, 2, 1)
    # now padded_embeddings is [N, C, T]

    second_pass_out = second_pass_model(padded_embeddings)

    # second_pass_out is of shape [N, C, T]
    second_pass_out = second_pass_out.permute(0, 2, 1)
    # now second_pass_out is of shape [N, T, C]

    if False:
        phone_seqs, _, _ = k2.ragged.unique_sequences(phone_seqs, True, True)
        phone_seqs = k2.ragged.remove_axis(phone_seqs, 0)
        phone_fsas = create_phone_fsas(phone_seqs)
        phone_fsas = k2.add_epsilon_self_loops(phone_fsas)

        probs = get_log_probs(phone_fsas, second_pass_out, len_per_path)

    second_pass_supervision_segments = torch.stack(
        (torch.arange(len_per_path.numel(), dtype=torch.int32),
         torch.zeros_like(len_per_path), len_per_path),
        dim=1)

    indices2 = torch.argsort(len_per_path, descending=True)
    second_pass_supervision_segments = second_pass_supervision_segments[
        indices2]
    # Note that path_to_seq is not changed!
    # No need to modify second_pass_out

    num_repeats_float = k2.ragged.RaggedFloat(
        num_repeats.shape(),
        num_repeats.values().to(torch.float32))
    path_weight = k2.ragged.normalize_scores(num_repeats_float,
                                             use_log=False).values

    second_pass_dense_fsa_vec = k2.DenseFsaVec(
        second_pass_out, second_pass_supervision_segments)

    second_pass_lattices = k2.intersect_dense_pruned(
        decoding_graph, second_pass_dense_fsa_vec, 20.0, 10.0, 300, 10000)

    # The number of FSAs in the second_pass_lattices may not
    # be equal to the number of paths since repeated paths are removed
    # by k2.ragged.unique_sequences

    inverted_indices2 = invert_permutation(indices2)

    second_pass_lattices = k2.index(
        second_pass_lattices,
        inverted_indices2.to(torch.int32).to(device))
    # now second_pass_lattices corresponds to the reordered paths
    # (due to k2.ragged.unique_sequences)

    if True:
        reordered_word_fsas = k2.index(word_fsas, new2old)

        reorded_lats = k2.compose(second_pass_lattices,
                                  reordered_word_fsas,
                                  treat_epsilons_specially=False)

        if reorded_lats.properties & fsa_properties.TOPSORTED_AND_ACYCLIC != fsa_properties.TOPSORTED_AND_ACYCLIC:
            reorded_lats = k2.top_sort(k2.connect(
                reorded_lats.to('cpu'))).to(device)

        # note some entries in `tot_scores_2nd_num` is -inf !!!
        tot_scores_2nd_num = reorded_lats.get_tot_scores(
            use_double_scores=True, log_semiring=True)

        #  for k in [0, 1, 2, 30, 40, 50]:
        #      pk, _ = k2.ragged.index(probs, torch.tensor([k],
        #                                                  dtype=torch.int32))
        #      assert pk.num_elements() == len_per_path[k]
        #      logging.info(
        #          f'\npath: {k}\ntot_scores: {tot_scores_2nd_num[k]}\nlog_probs:{str(pk)}'
        #      )

        tot_scores_2nd_den = second_pass_lattices.get_tot_scores(
            log_semiring=True, use_double_scores=use_double_scores)

        tot_scores_2nd = tot_scores_2nd_num - tot_scores_2nd_den
        logging.info(f'num: {tot_scores_2nd_num}')
        logging.info(f'den: {tot_scores_2nd_den}')
        logging.info(f'2nd: {tot_scores_2nd}')

        #  print(
        #      'word',
        #      reordered_word_fsas.arcs.row_splits(1)[1:] -
        #      reordered_word_fsas.arcs.row_splits(1)[:-1])
        #  print(
        #      reorded_lats.arcs.row_splits(1)[1:] -
        #      reorded_lats.arcs.row_splits(1)[:-1])
        #  print('2 num', tot_scores_2nd_num)
        #  print('2 den', tot_scores_2nd_den)
        #  print('2 ', tot_scores_2nd)

        #  import sys
        #  sys.exit(0)
    else:
        tot_scores_2nd = second_pass_lattices.get_tot_scores(
            use_double_scores=True, log_semiring=True)
        #  logging.info(f'tot_scores_2nd: {tot_scores_2nd}')

    # Now tot_scores_2nd[i] corresponds to sorted_path_i
    # `sorted` here is due to k2.ragged.unique_sequences.
    # We have to use `new2old` to map it to the original unsorted path

    # Note that path_weight was not reordered

    # argmax for the 1st pass
    ragged_tot_scores_1st = k2.RaggedFloat(seq_to_path_shape,
                                           tot_scores_1st.to(torch.float32))
    argmax_indexes_1st = k2.ragged.argmax_per_sublist(ragged_tot_scores_1st)
    argmax_indexes_1st = torch.clamp(argmax_indexes_1st, min=0)

    #  logging.info(f'1st: {tot_scores_1st}')
    tot_scores = tot_scores_1st
    tot_scores[new2old.long()] += tot_scores_2nd * path_weight
    ragged_tot_scores = k2.RaggedFloat(seq_to_path_shape,
                                       tot_scores.to(torch.float32))
    _argmax_indexes = k2.ragged.argmax_per_sublist(ragged_tot_scores)
    #  logging.info(f'indexes: {argmax_indexes}')
    #  print(argmax_indexes)
    # argmax_indexes may contain -1. This case happens
    # when a sublist contains all -inf
    argmax_indexes = torch.clamp(_argmax_indexes, min=0)
    if argmax_indexes.sum() != _argmax_indexes.sum():
        logging.info(
            f'-1 appears: {tot_scores}, {ragged_tot_scores}, {_argmax_indexes}'
        )

    #  logging.info(f'\n{argmax_indexes_1st}\n{argmax_indexes}')

    paths = k2.ragged.remove_axis(paths, 0)

    best_paths = k2.index(paths, argmax_indexes)
    labels = k2.index(lats.labels.contiguous(), best_paths)
    aux_labels = k2.index(lats.aux_labels, best_paths.values())
    labels = k2.ragged.remove_values_eq(labels, -1)
    best_paths = k2.linear_fsa(labels)
    best_paths.aux_labels = aux_labels

    return best_paths
