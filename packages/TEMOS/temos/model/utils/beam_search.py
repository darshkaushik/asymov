import torch
import torch.utils.data as tud
from tqdm import tqdm
from torch.nn import Module, Transformer as T
# from tools import remove_padding
import pdb


def remove_padding(tensors, lengths):
    return [tensor[:tensor_length] for tensor, tensor_length in zip(tensors, lengths)]

# dummy wrapper to make the usage clear
def remove_padding_and_EOS(tensors, lengths):
    return remove_padding(tensors, lengths)


def beam_search(model, sequences, predictions=20, beam_width=5, batch_size=16):
    """
    Note to Darsh: function implements Beam Search for diverse sequences in motion words
    (this one is simpler for sequence to sequence.
    Creating one with word maps as a better alternative, although it needs <start> and <end> ids for the tree.)

    The method can compute several outputs in parallel with the first dimension of sequences.
    Parameters
    ----------
    sequences: Tensor of shape (examples, length)
        The sequences to start the decoding process. (treating this as <start> for now.)
    predictions: int
        The number of tokens to stitch to sequences. Also the number of splits in the tree.
    beam_width: int
        The number of candidates to keep in the search.
    batch_size: int
        The batch size of the method.
    Returns
    -------
    sequences: Tensor of shape (examples, length + predictions)

    probabilities: FloatTensor of size examples
        The estimated log-probabilities for the output sequences.
    """
    with torch.no_grad():

        next_probabilities, next_latent, next_distribution = model.motion_to_motion_forward(sequences)[:, -1, :]
        #Using the motion_to_motion forward function from asymov

        vocabulary_size = next_probabilities.shape[-1]
        probabilities, idx = next_probabilities.squeeze().log_softmax(-1) \
            .topk(k=beam_width, axis=-1)
        sequences = sequences.repeat((beam_width, 1, 1)).transpose(0, 1) \
            .flatten(end_dim=-2)
        next_chars = idx.reshape(-1, 1)
        sequences = torch.cat((sequences, next_chars), axis=-1)

        predictions_iterator = range(predictions - 1) #one prediction already done before for loop

        for i in predictions_iterator:
            dataset = tud.TensorDataset(sequences)
            loader = tud.DataLoader(dataset, batch_size=batch_size)
            next_probabilities = []
            iterator = iter(loader)

            for (x,) in iterator:
                probabilities_i, latent_i, distribution_i = model.motion_to_motion_forward(x)[:, -1, :].log_softmax(-1)
                next_probabilities.append(probabilities_i)
            next_probabilities = torch.cat(next_probabilities, axis=0)
            next_probabilities = next_probabilities.reshape(
                (-1, beam_width, next_probabilities.shape[-1])
            )
            probabilities = probabilities.unsqueeze(-1) + next_probabilities
            probabilities = probabilities.flatten(start_dim=1)
            probabilities, idx = probabilities.topk(
                k=beam_width,
                axis=-1
            )
            next_chars = torch.remainder(idx, vocabulary_size).flatten() \
                .unsqueeze(-1)
            best_candidates = (idx / vocabulary_size).long()
            best_candidates += torch.arange(sequences.shape[0] // beam_width,device=sequences.device).unsqueeze(-1) * beam_width
            sequences = sequences[best_candidates].flatten(end_dim=-2)
            sequences = torch.cat((sequences, next_chars), axis=1)
        return sequences.reshape(-1, beam_width, sequences.shape[-1]), probabilities


def beam_search_nat(model, memory, beam_size, src_mask, max_len=256, start=0, end=1):
    assert beam_size > 1
    finished = torch.zeros(1, dtype=torch.bool)
    paths = torch.full((1, max_len + 1), start)
    probs = torch.zeros(1)

    for i in range(1, max_len + 1):
        mask = torch.triu(torch.ones((1, i,i)), diagonal=1)==0
        logits = model.decode(memory.expand((~finished).count_nonzero(), -1, -1),
            src_mask, paths[~finished, :i], mask)
        print(len(logits), len(logits[0]))
        scores = probs[~finished].unsqueeze(1) + model.generator(logits[:, -1])
        print(len(scores))
        if i == 1: # increase capacity to beam_size
            finished = finished.repeat(beam_size)
            paths = paths.repeat(beam_size, 1)
            probs = probs.repeat(beam_size)

        candidates = paths[~finished]
        topv, topi = torch.topk(scores.flatten(), beam_size)
        if any(finished): # length normalization
            for j in range(beam_size):
                finished[finished.nonzero(as_tuple=True)] ^= probs[finished] < (topv[j] / i)
            if (~finished).count_nonzero() > beam_size:
                beam_size = (~finished).sum()
                topv, topi = torch.topk(scores.flatten(), beam_size)

        paths[~finished] = candidates[
            torch.div(topi, model.tgt_vocab_size, rounding_mode='trunc')
        ]
        paths[~finished, i] = topi % model.tgt_vocab_size
        probs[~finished] = topv

        finished |= paths[:, i] == end
        beam_size = (~finished).count_nonzero()
        probs[paths[:, i] == end] /= i
        if all(finished): break

    best_path = paths[probs.argmax()]
    end_index = (best_path == end).nonzero()
    return best_path[1:end_index] if end_index.numel() else best_path[1:]


def beam_search_auto(                       #alpha
    model,
    src,
    tgt,
    src_mask,
    src_padding_mask,
    end_symbol: int,
    max_len = 220,
    batch_size = 128,            #passing this is important!
    beam_width = 5,
    traj: bool = True,
):

    with torch.no_grad():
        # src = src.repeat((1, beam_width))
        src = src.repeat_interleave(beam_width, dim=1)
        # src_padding_mask = src_padding_mask.repeat((beam_width,1))
        src_padding_mask = src_padding_mask.repeat_interleave(beam_width, dim=0)

        tgt = tgt.repeat((1,beam_width))
        tgt_padding_mask = tgt.new_full((batch_size*beam_width, 1), False, dtype=torch.bool)  # [Batch Size, 1], 1 as for 1st frame
        tgt_mask = (T.generate_square_subsequent_mask(tgt.size(0))      #size(0) for num of decoded
                          .to(tgt.device, dtype=torch.bool))
        tgt_len = tgt.new_full((batch_size*beam_width, ), max_len)

        memory = model.encode(src, src_mask, src_padding_mask)  # [Frames, Batches, *]

        if traj:
          tgt_traj = src.new_zeros((1, batch_size*beam_width, 3), dtype=torch.long) #[1, Batch size, 3], 1 as for 1st frame

      #first decoding
          out = model.decode(tgt, memory, tgt_mask, None, tgt_padding_mask, src_padding_mask, tgt_traj = tgt_traj) #[Frames, Batch Size, *]
          next_root = model.traj_generator(out[-1]) #[Batch Size, 3]
          tgt_traj = torch.cat([tgt_traj, next_root.unsqueeze(0)]) #[Frames+1, Batch size, 3]

        else:
          out = model.decode(tgt, memory, tgt_mask, None, tgt_padding_mask, src_padding_mask)  # [Frames, Batch Size, *]

        logits = model.generator(out[-1])
        next_probabilities = logits#[-1, :]
        probabilities, next_chars = next_probabilities.squeeze().log_softmax(-1).topk(k=beam_width, axis=-1)
        next_tokens = next_chars[torch.arange(0, next_chars.shape[0], beam_width),:].reshape(-1)

        tgt = torch.cat((tgt, next_tokens.unsqueeze(0)))

        tgt_len = torch.where(torch.logical_and(next_tokens==end_symbol, tgt_len==max_len), 0, tgt_len)     #0: i(th) decoding

        # TODO change to range(0 or 1, max_len)
        for i in tqdm(range(1,max_len), "beam autoregressive translation", None):

          tgt_mask = (T.generate_square_subsequent_mask(tgt.size(0))      #size(0) for num of decoded
                          .to(tgt.device, dtype=torch.bool))                      #tokens.
          tgt_padding_mask = torch.cat([tgt_padding_mask, (tgt_len<=i).unsqueeze(-1)], dim=-1)

          if traj:
              out = model.decode(tgt, memory, tgt_mask, None, tgt_padding_mask, src_padding_mask, tgt_traj = tgt_traj) #[Frames, Batch Size, *]
          else:
              out = model.decode(tgt, memory, tgt_mask, None, tgt_padding_mask, src_padding_mask)  # [Frames, Batch Size, *]

          logits = model.generator(out[-1])
          vocabulary_size = logits.shape[1]
          probabilities, next_chars = logits.reshape(batch_size, -1).squeeze().log_softmax(-1).topk(k=beam_width, axis=-1)

          next_tokens = torch.remainder(next_chars, vocabulary_size)#.transpose(1,0)#.flatten().unsqueeze(-1)
          best_candidates = (next_chars / vocabulary_size).long()

          #sorting/selecting beams acc to
          reshape = (beam_width,-1) if batch_size==1 else (batch_size, beam_width,-1)
          expand = (-1,tgt.shape[0]) if batch_size==1 else (-1,-1,tgt.shape[0])
          sorted_batch = torch.gather(tgt.T.unsqueeze(-1).reshape(reshape),
                                      dim=-2,
                                      index=best_candidates.unsqueeze(-1).expand(expand))
          # torch.gather(sorted_batch,dim=-2,index=best_candidates.unsqueeze(-1).expand(expand))
          tgt_temp = sorted_batch.reshape(batch_size*beam_width,tgt.shape[0]).T

          tgt = torch.cat(( tgt_temp, next_tokens.reshape(-1).unsqueeze(0) ))

          if traj:
            next_root = model.traj_generator(out[-1]) #[Batch Size, 3]

            expand_traj = (-1,tgt_traj.shape[0],3) if batch_size==1 else (-1,-1,tgt_traj.shape[0],3)
            reshape_traj = (beam_width,-1,3) if batch_size==1 else (batch_size, beam_width,-1,3)
            sorted_traj = torch.gather(tgt_traj.permute(1,0,2).unsqueeze(-2).reshape(reshape_traj),
                          dim=-3,
                          index=best_candidates.unsqueeze(-1).unsqueeze(-1).expand(expand_traj))
            # torch.gather(tgt_traj.permute(1,0,2).unsqueeze(-2).reshape(reshape_traj),dim=-3,index=best_candidates.unsqueeze(-1).unsqueeze(-1).expand(expand_traj))
            tgt_traj_temp = sorted_traj.reshape(batch_size*beam_width, tgt_traj.shape[0], 3).permute(1,0,2)

            tgt_traj = torch.cat([tgt_traj_temp, next_root.unsqueeze(0)]) #[Frames+1, Batch size, 3]

          #TODO store effective length without EOS/BOS
          tgt_len = torch.where(torch.logical_and(next_tokens.reshape(-1)==end_symbol, tgt_len==max_len), i, tgt_len)     #this requires debugging

        #TODO remove BOS and EOS (change remove_padding function)
        tgt_list =  remove_padding_and_EOS(tgt[1:].permute(1, 0), tgt_len)

        if traj:
          #TODO remove BOS and EOS (change remove_padding function)
          tgt_traj_list =  remove_padding_and_EOS(tgt_traj[1:].permute(1, 0, 2), tgt_len)
          return tgt_list, tgt_traj_list #Tuple[List[Tensor[Frames]]]

        return tgt_list


def beam_search_unit(                       #alpha
    model,
    src,
    tgt,
    src_mask,
    src_padding_mask,
    end_symbol: int,
    max_len = 220,
    beam_width = 5,
    traj: bool = True,
):
    with torch.no_grad():
        # src = src.repeat((1, beam_width))
        src = src.repeat_interleave(beam_width, dim=1)
        # src_padding_mask = src_padding_mask.repeat((beam_width,1))
        src_padding_mask = src_padding_mask.repeat_interleave(beam_width, dim=0)

        tgt = tgt.repeat((1,beam_width))
        tgt_padding_mask = tgt.new_full((beam_width, 1), False, dtype=torch.bool)  # [Batch Size, 1], 1 as for 1st frame
        tgt_mask = (T.generate_square_subsequent_mask(tgt.size(0))      #size(0) for num of decoded
                          .to(tgt.device, dtype=torch.bool))
        tgt_len = tgt.new_full((beam_width, ), max_len)

        memory = model.encode(src, src_mask, src_padding_mask)  # [Frames, Batches, *]

        if traj:
          tgt_traj = src.new_zeros((1, beam_width, 3), dtype=torch.long) #[1, Batch size, 3], 1 as for 1st frame

      #first decoding
          out = model.decode(tgt, memory, tgt_mask, None, tgt_padding_mask, src_padding_mask, tgt_traj = tgt_traj) #[Frames, Batch Size, *]
          next_root = model.traj_generator(out[-1]) #[Batch Size, 3]
          tgt_traj = torch.cat([tgt_traj, next_root.unsqueeze(0)]) #[Frames+1, Batch size, 3]

        else:
          out = model.decode(tgt, memory, tgt_mask, None, tgt_padding_mask, src_padding_mask)  # [Frames, Batch Size, *]

        logits = model.generator(out[-1])
        next_probabilities = logits#[-1, :]
        probabilities, next_chars = next_probabilities.squeeze().log_softmax(-1).topk(k=beam_width, axis=-1)
        next_tokens = next_chars[torch.arange(0, next_chars.shape[0], beam_width),:].reshape(-1)

        tgt = torch.cat((tgt, next_tokens.unsqueeze(0)))

        tgt_len = torch.where(torch.logical_and(next_tokens==end_symbol, tgt_len==max_len), 0, tgt_len)     #0: i(th) decoding

        for i in tqdm(range(1,max_len), "beam autoregressive translation", None):

          tgt_mask = (T.generate_square_subsequent_mask(tgt.size(0))      #size(0) for num of decoded
                          .to(tgt.device, dtype=torch.bool))                      #tokens.
          tgt_padding_mask = torch.cat([tgt_padding_mask, (tgt_len<=i).unsqueeze(-1)], dim=-1)

          if traj:
              out = model.decode(tgt, memory, tgt_mask, None, tgt_padding_mask, src_padding_mask, tgt_traj = tgt_traj) #[Frames, Batch Size, *]
          else:
              out = model.decode(tgt, memory, tgt_mask, None, tgt_padding_mask, src_padding_mask)  # [Frames, Batch Size, *]

          logits = model.generator(out[-1])
          vocabulary_size = logits.shape[1]
          probabilities, next_chars = logits.reshape(-1).squeeze().log_softmax(-1).topk(k=beam_width, axis=-1)

          next_tokens = torch.remainder(next_chars, vocabulary_size)#.transpose(1,0)#.flatten().unsqueeze(-1)
          best_candidates = (next_chars / vocabulary_size).long()

          #sorting/selecting beams acc to best_candidates
          sorted_batch = torch.gather(tgt.T.unsqueeze(-1).reshape(beam_width,-1),
                                      dim=0,
                                      index=best_candidates.unsqueeze(-1).expand(-1,tgt.shape[0]))
          tgt_temp = sorted_batch.reshape(beam_width,tgt.shape[0]).T

          tgt = torch.cat(( tgt_temp, next_tokens.reshape(-1).unsqueeze(0) ))

          if traj:
            next_root = model.traj_generator(out[-1]) #[Batch Size, 3]

            sorted_traj = torch.gather(tgt_traj.permute(1,0,2).unsqueeze(-2).reshape(beam_width,-1,3),
                          dim=0,
                          index=best_candidates.unsqueeze(-1).unsqueeze(-1).expand(-1,tgt_traj.shape[0],3))
            tgt_traj_temp = sorted_traj.reshape(beam_width, tgt_traj.shape[0], 3).permute(1,0,2)

            tgt_traj = torch.cat([tgt_traj_temp, next_root.unsqueeze(0)]) #[Frames+1, Batch size, 3]

          tgt_len = torch.where(torch.logical_and(next_tokens.reshape(-1)==end_symbol, tgt_len==max_len), i, tgt_len)     #this requires debugging

        tgt_list =  remove_padding(tgt[1:].permute(1, 0), tgt_len)

        if traj:
          tgt_traj_list =  remove_padding(tgt_traj[1:].permute(1, 0, 2), tgt_len)
          return tgt_list, tgt_traj_list #Tuple[List[Tensor[Frames]]]

        return tgt_list


def diverse_beam_search_auto(                       #alpha
    model,
    src,
    tgt,
    src_mask,
    src_padding_mask,
    end_symbol: int,
    max_len = 220,
    batch_size = 128,            #CHECK: this batch size is different from the number of sequences passed
    beam_width = 5,
    traj: bool = True,
):
    with torch.no_grad():

        src = src.repeat((1, beam_width))
        # src = src.repeat_interleave(beam_width, dim=1)
        src_padding_mask = src_padding_mask.repeat((beam_width,1))
        # src_padding_mask = src_padding_mask.repeat_interleave(beam_width, dim=0)

        tgt = tgt.repeat((1,beam_width))
        tgt_padding_mask = tgt.new_full((batch_size*beam_width, 1), False, dtype=torch.bool)  # [Batch Size, 1], 1 as for 1st frame
        # tgt_padding_mask = tgt_padding_mask.repeat((beam_width, 1))
        tgt_len = tgt.new_full((batch_size*beam_width, ), max_len)
        # tgt_len = tgt_len.repeat(beam_width, )

        if traj:
            tgt_traj = src.new_zeros((1, batch_size*beam_width, 3), dtype=torch.long) #[1, Batch size, 3], 1 as for 1st frame

        memory = model.encode(src, src_mask, src_padding_mask)  # [Frames, Batches, *]

        for i in tqdm(range(0,max_len), "diverse autoregressive translation", None):

            tgt_mask = (T.generate_square_subsequent_mask(tgt.size(0))      #size(0) for num of decoded
                    .to(tgt.device, dtype=torch.bool))                      #tokens.
            out = model.decode(tgt, memory, tgt_mask, None, tgt_padding_mask, src_padding_mask, tgt_traj = tgt_traj) #[Frames, Batch Size, *]
            if traj:
                next_root = model.traj_generator(out[-1]) #[Batch Size, 3]
                tgt_traj = torch.cat([tgt_traj, next_root.unsqueeze(0)]) #[Frames+1, Batch size, 3]
            else:
                out = model.decode(tgt, memory, tgt_mask, None, tgt_padding_mask, src_padding_mask)  # [Frames, Batch Size, *]
            logits = model.generator(out[-1])
            next_probabilities = logits#[-1, :]

            probabilities, next_chars = next_probabilities.squeeze().log_softmax(-1).topk(k=beam_width, axis=-1)

            next_token_mask = torch.zeros(next_chars.shape, dtype=torch.bool)
            next_token_mask[:batch_size, 0] = True             #first beam decoded acc to highest prob (normal beam search)

            for i in range(batch_size, next_chars.shape[0], batch_size):
                mask_num = (next_chars[i:i+batch_size][:, None] == next_chars[next_token_mask].reshape(-1,batch_size).T.unsqueeze(-1)).any(dim=1).long()
                unique_token_idx = torch.argmin(mask_num, axis=1)

                assert len(unique_token_idx) == batch_size

                next_token_mask[torch.arange(i, i+batch_size), unique_token_idx] = True

            next_tokens = next_chars[next_token_mask]#.reshape(-1,batch_size).T.reshape(-1)
            next_tokens_prob = probabilities[next_token_mask]

            assert len(next_tokens) == batch_size*beam_width, f'''Decoded T(th) tokens not equal to batch*beam size, required{batch_size*beam_width}
                                                                                                                     current shape: {len(next_tokens)}'''

            tgt = torch.cat((tgt, next_tokens.unsqueeze(0)))

            tgt_len = torch.where(torch.logical_and(next_tokens==end_symbol, tgt_len==max_len), i, tgt_len)     #tNote to darsh
            tgt_padding_mask = torch.cat([tgt_padding_mask, (tgt_len<=i).unsqueeze(-1)], dim=-1)

        # last_probs = probabilities[next_token_mask].reshape(-1,batch_size)
        # top_b = torch.argmax(last_probs[1:], axis=0)      #like top-G, but gives top beams for batch elements correspondingly
        #                                                   #removed first beam as its greedy
        # last_prob_mask = torch.zeros(last_probs[1:].shape, dtype=torch.bool)
        # last_prob_mask[top_b, torch.arange(batch_size)] = True

        # final_unordered = tgt[:, batch_size:][:, last_prob_mask.reshape(-1)]              #beams with highest prob, for all batch elements
        # reorder = (last_prob_mask.reshape(-1).long().nonzero()%batch_size).squeeze()      #reorder batch elements

        # tgt_len = tgt_len[batch_size:][last_prob_mask.reshape(-1)][reorder]
        # tgt_list=  remove_padding(final_unordered.T[reorder], tgt_len)
        assert tgt.shape[0] == max_len+1, f"At this point, frames should be <start> frame + max_decoded: {1} + {max_len}"

        tgt = tgt[1:]
        tgt = tgt.T.reshape(beam_width, -1).T.reshape(-1, tgt.shape[0], beam_width).permute(0,2,1).reshape(-1, tgt.shape[0])
        tgt_list = remove_padding_and_EOS(tgt, tgt_len)

        tmp = torch.tensor([len(x) for x in tgt_list]).to(tgt_len.device)
        assert torch.all(tmp == tgt_len), f'''Len of each tensor does not match tgt_len after BOS, pad,
                            EOS removal for {torch.nonzero(torch.logical_not(torch.tensor([len(x) for x in tgt_list]) == tgt_len)).squeeze()}'''

        assert len(tgt_list) == beam_width*batch_size, f'''tgt_list size not equal to beam*batch, required: {beam_width*batch_size}
                                                                                                  current shape: {len(tgt_list)}'''

        if traj:
            tgt_traj = tgt_traj[1:]
            tgt_traj = tgt_traj.permute(1,0,2).reshape(beam_width, -1, 3).permute(1,0,2).reshape(-1, tgt_traj.shape[0], beam_width, 3).permute(0, 2, 1, 3).reshape(-1, tgt_traj.shape[0], 3)
            # final_unordered_traj = tgt_traj[:, batch_size:][:, last_prob_mask.reshape(-1)]
            # tgt_traj_list =  remove_padding(final_unordered_traj.permute(1, 0, 2)[reorder], tgt_len)
            tgt_traj_list =  remove_padding_and_EOS(tgt_traj, tgt_len)

            tmp = torch.tensor([len(x) for x in tgt_traj_list]).to(tgt_len.device)
            assert torch.all(tmp == tgt_len), f'''Len of each tensor does not match tgt_len after BOS, pad,
                            EOS removal for {torch.nonzero(torch.logical_not(torch.tensor([len(x) for x in tgt_traj_list]) == tgt_len)).squeeze()}'''

            assert len(tgt_traj_list) == beam_width*batch_size, f'''tgt_list size not equal to beam*batch, required: {beam_width*batch_size}
                                                                                                           current shape: {len(tgt_traj_list)}'''

            return tgt_list, tgt_traj_list #Tuple[List[Tensor[Frames]]]

        return tgt_list



def diverse_beam_search_unit(                       #alpha
    model,
    src,
    tgt,
    src_mask,
    src_padding_mask,
    end_symbol: int,
    max_len = 220,
    beam_width = 5,
    traj: bool = True,
):
    with torch.no_grad():               ##
        src = src.repeat((1, beam_width))
        # src = src.repeat_interleave(beam_width, dim=1)
        src_padding_mask = src_padding_mask.repeat((beam_width,1))
        # src_padding_mask = src_padding_mask.repeat_interleave(beam_width, dim=0)

        tgt = tgt.repeat((1,beam_width))
        tgt_padding_mask = tgt.new_full((beam_width, 1), False, dtype=torch.bool)  # [Batch Size, 1], 1 as for 1st frame
        # tgt_padding_mask = tgt_padding_mask.repeat((beam_width, 1))
        tgt_len = tgt.new_full((beam_width, ), max_len)
        # tgt_len = tgt_len.repeat(beam_width, )

        if traj:
            tgt_traj = src.new_zeros((1, beam_width, 3), dtype=torch.long) #[1, Batch size, 3], 1 as for 1st frame

        memory = model.encode(src, src_mask, src_padding_mask)  # [Frames, Batches, *]

        for i in tqdm(range(0,max_len), "diverse autoregressive translation", None):

            tgt_mask = (T.generate_square_subsequent_mask(tgt.size(0))      #size(0) for num of decoded
                    .to(tgt.device, dtype=torch.bool))                      #tokens.
            out = model.decode(tgt, memory, tgt_mask, None, tgt_padding_mask, src_padding_mask, tgt_traj = tgt_traj) #[Frames, Batch Size, *]
            if traj:
                next_root = model.traj_generator(out[-1]) #[Batch Size, 3]
                tgt_traj = torch.cat([tgt_traj, next_root.unsqueeze(0)]) #[Frames+1, Batch size, 3]
            else:
                out = model.decode(tgt, memory, tgt_mask, None, tgt_padding_mask, src_padding_mask)  # [Frames, Batch Size, *]

            logits = model.generator(out[-1])
            next_probabilities = logits#[-1, :]

            probabilities, next_chars = next_probabilities.squeeze().log_softmax(-1).topk(k=beam_width, axis=-1)

            next_token_mask = next_chars.new_zeros(next_chars.shape, dtype=torch.bool)
            next_token_mask[0, 0] = True             #first beam decoded acc to highest prob (normal beam search)

            for i in range(1,next_chars.shape[0]):
                # mask_num = (next_chars[i][:, None] == next_chars[next_token_mask].unsqueeze(-1)).any(dim=1).long()#.reshape(-1,batch_size).T.unsqueeze(-1)).any(dim=1).long()
                mask_num = (next_chars[i][None, :] == next_chars[next_token_mask].unsqueeze(-1).T.unsqueeze(-1)).any(dim=1).long()#.reshape(-1,batch_size).T.unsqueeze(-1)).any(dim=1).long()
                unique_token_idx = torch.argmin(mask_num)
                next_token_mask[i, unique_token_idx] = True

            next_tokens = next_chars[next_token_mask]#.reshape(-1,batch_size).T.reshape(-1)
            next_tokens_prob = probabilities[next_token_mask]

            tgt = torch.cat((tgt, next_tokens.unsqueeze(0)))

            tgt_len = torch.where(torch.logical_and(next_tokens==end_symbol, tgt_len==max_len), i, tgt_len)     #tNote to darsh
            tgt_padding_mask = torch.cat([tgt_padding_mask, (tgt_len<=i).unsqueeze(-1)], dim=-1)

        tgt_list =  remove_padding_and_EOS(tgt[1:].permute(1, 0), tgt_len)

        if traj:
            # final_unordered_traj = tgt_traj[:, batch_size:][:, last_prob_mask.reshape(-1)]
            # tgt_traj_list =  remove_padding(final_unordered_traj.permute(1, 0, 2)[reorder], tgt_len)
            tgt_traj_list =  remove_padding_and_EOS(tgt_traj[1:].permute(1, 0, 2), tgt_len)
            return tgt_list, tgt_traj_list #Tuple[List[Tensor[Frames]]]

        return tgt_list




# def diverse_beam_search_auto(                       #alpha
#     model,
#     src,
#     tgt,
#     src_mask,
#     src_padding_mask,
#     tgt_mask,
#     tgt_padding_mask,
#     end_symbol: int,
#     max_len = 220,
#     beam_width = 5,
#     batch_size = 128            #CHECK: this batch size is different from the number of sequences passed
# ):


#     with torch.no_grad():

#         memory = model.encode(src, src_mask, src_padding_mask)  # [Frames, Batches, *]
#         out = model.decode(tgt, memory, tgt_mask, None, tgt_padding_mask, src_padding_mask)  # [Frames, Batch Size, *]
#         logits = model.generator(out[-1])

#         next_probabilities = logits#[-1, :]
#         # vocabulary_size = next_probabilities.shape[-1]
#         probabilities, next_chars = next_probabilities.squeeze().log_softmax(-1).topk(k=beam_width, axis=-1)
#         tgt = tgt.repeat((beam_width, 1))       #repeat BOS  for beam width
#         tgt_padding_mask = tgt_padding_mask.unsqueeze(0).repeat((beam_width, 1, 1))

#         tgt_len = tgt.new_full((beam_width, batch_size), max_len)                              #[beam_width, Batch Size] #same dtype as tgt

#         tgt = torch.cat((tgt.unsqueeze(-2), next_chars.transpose(1,0).unsqueeze(-2)), -2)      #concat next tokens for each beam (vertically)

#         diverse_tokens = torch.tensor([])               # to store decoded tokens in diverse fashion
#         seq_prob = torch.tensor([])                     # and corresponding probabilities in the last decoding step

#         predictions_iterator = range(1, max_len - 1)     #1 (0th) prediction already done; max_len - 1 bec <start_id> -> 1
#         for i in predictions_iterator:
#             tgt_mask = (T.generate_square_subsequent_mask(tgt.size(1))      #size(1) for num of decoded
#                     .to(tgt.device, dtype=torch.bool))                      #tokens. 0th is beam size

#             tgt_padding_mask = torch.cat([tgt_padding_mask, (tgt_len<=i).unsqueeze(-1)], dim=-1)

#             for b in range(beam_width):         #if using i, it's continued outside scope of for loop
#                 out_temp = model.decode(tgt[b], memory, tgt_mask, None, tgt_padding_mask[b], src_padding_mask)
#                 logits_temp = model.generator(out_temp[-1])

#                 if b == 0:
#                   probabilities, idx = logits_temp.log_softmax(-1).topk(k=1, axis=-1)    ##
#                   diverse_tokens = idx.squeeze().unsqueeze(0)
#                   if i == max_len - 1:                                                   #last decoded step contains final multiplied probabilities of all tokens
#                     seq_prob = probabilities.squeeze().unsqueeze(0)

#                 else:
#                   probabilities, idx = logits_temp.log_softmax(-1).topk(k=b+1, axis=-1)

#                   unique_mask = torch.stack([(diverse_tokens == idx.T[i]).any(dim=0) for i in range(idx.T.shape[0])], dim=1).long()
#                   first_unique = unique_mask.argmin(1)                                #in all elements of batch
#                   # complete_index = torch.stack([torch.arange(unique_mask.shape[0]), first_unique], dim=1)   #containes coordinates -> diverse tokens, dim=0 -> batch_size

#                   unique_tokens = idx[torch.arange(first_unique.shape[0]), first_unique]      #needs to be checked
#                   current_token_prob = probabilities[torch.arange(first_unique.shape[0]), first_unique]

#                   diverse_tokens = torch.cat((diverse_tokens, unique_tokens.unsqueeze(0)))
#                   if i == max_len - 1:
#                     seq_prob = torch.cat((seq_prob, current_token_prob.unsqueeze(0)))

#             tgt_len = torch.where(torch.logical_and(next_chars==end_symbol, tgt_len==max_len), i+2, tgt_len)
#             tgt = torch.cat((tgt, diverse_tokens.unsqueeze(-2)), -2)

#         top_b = torch.argmax(seq_prob[1:], axis=0)        #like top-G, but gives top beams for batch elements correspondingly, acc. to prob
#         # pdb.set_trace()
#         return tgt[top_b, :, torch.arange(tgt.size(-1))]
