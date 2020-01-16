#!/bin/env python
# Copyright 2018 Hitachi, Ltd. (author: Yusuke Fujita)
#
"""
 This module computes features
"""

import numpy as np
import librosa

def transform(
        Y,
        transform_type=None,
        dtype=np.float32):
    """ Transform STFT feature

    Args:
        Y: STFT
            (n_frames, n_bins)-shaped np.complex array
        transform_type:
            None, "log"
        dtype: output data type
            np.float32 is expected
    Returns:
        Y (numpy.array): transformed feature
    """
    Y = np.abs(Y)
    if transform_type == 'log':
        Y = np.log(np.maximum(Y, 1e-06))
    return Y.astype(dtype)


def splice(Y, context_size=0):
    """ Frame splicing

    Args:
        Y: feature
            (n_frames, n_featdim)-shaped numpy array
        context_size:
            number of frames concatenated on left-side
            if context_size = 5, 11 frames are concatenated.

    Returns:
        Y_spliced: spliced feature
            (n_frames, n_featdim * (2 * context_size + 1))-shaped
    """
    Y_pad = np.pad(
            Y,
            [(context_size, context_size), (0, 0)],
            'constant')
    Y_spliced = np.lib.stride_tricks.as_strided(
            Y_pad,
            (Y.shape[0], Y.shape[1] * (2 * context_size + 1)),
            (Y.itemsize * Y.shape[1], Y.itemsize), writeable=False)
    return Y_spliced


def stft(
        data,
        frame_size=1024,
        frame_shift=256):
    """ Compute STFT features

    Args:
        data: audio signal
            (n_samples,)-shaped np.float32 array
        frame_size: number of samples in a frame (must be a power of two)
        frame_shift: number of samples between frames

    Returns:
        stft: STFT frames
            (n_frames, n_bins)-shaped np.complex64 array
    """
    # HACK: The last frame is ommited
    #       as librosa.stft produces such an excessive frame
    if len(data) % frame_shift == 0:
        return librosa.stft(data, n_fft=frame_size,
                            hop_length=frame_shift).T[:-1]
    else:
        return librosa.stft(data, n_fft=frame_size, hop_length=frame_shift).T


def _count_frames(data_len, size, shift):
    # HACK: Assuming librosa.stft(..., center=True)
    n_frames = 1 + int(data_len / shift)
    if data_len % shift == 0:
        n_frames = n_frames - 1
    return n_frames


def get_frame_labels(
        kaldi_obj,
        rec,
        start=0,
        end=None,
        frame_size=1024,
        frame_shift=256,
        n_speakers=None):
    """ Get frame-aligned labels of given recording
    Args:
        kaldi_obj (KaldiData)
        rec (str): recording id
        start (int): start frame index
        end (int): end frame index
            None means the last frame of recording
        frame_size (int): number of frames in a frame
        frame_shift (int): number of shift samples
        n_speakers (int): number of speakers
            if None, the value is given from data
    Returns:
        T: label
            (n_frames, n_speakers)-shaped np.int32 array
    """
    filtered_segments = kaldi_obj.segments[kaldi_obj.segments['rec'] == rec]
    speakers = np.unique(
            [kaldi_obj.utt2spk[seg['utt']] for seg
                in filtered_segments]).tolist()
    if n_speakers is None:
        n_speakers = len(speakers)
    es = end * frame_shift if end is not None else None
    data, rate = kaldi_obj.load_wav(
            rec, start * frame_shift, es)
    n_frames = _count_frames(len(data), frame_size, frame_shift)
    T = np.zeros((n_frames, n_speakers), dtype=np.int32)
    if end is None:
        end = n_frames

    for seg in filtered_segments:
        speaker_index = speakers.index(kaldi_obj.utt2spk[seg['utt']])
        start_frame = np.rint(
                seg['st'] * rate / frame_shift).astype(int)
        end_frame = np.rint(
                seg['et'] * rate / frame_shift).astype(int)
        rel_start = rel_end = None
        if start <= start_frame and start_frame < end:
            rel_start = start_frame - start
        if start < end_frame and end_frame <= end:
            rel_end = end_frame - start
        if rel_start is not None or rel_end is not None:
            T[rel_start:rel_end, speaker_index] = 1
    return T


def get_labeledSTFT(
        kaldi_obj,
        rec, start, end, frame_size, frame_shift,
        n_speakers=None):
    """ Extracts STFT and corresponding labels

    Extracts STFT and corresponding diarization labels for
    given recording id and start/end times

    Args:
        kaldi_obj (KaldiData)
        rec (str): recording id
        start (int): start frame index
        end (int): end frame index
        frame_size (int): number of samples in a frame
        frame_shift (int): number of shift samples
        n_speakers (int): number of speakers
            if None, the value is given from data
    Returns:
        Y: STFT
            (n_frames, n_bins)-shaped np.complex64 array,
        T: label
            (n_frmaes, n_speakers)-shaped np.int32 array.
    """
    data, rate = kaldi_obj.load_wav(
            rec, start * frame_shift, end * frame_shift)
    Y = stft(data, frame_size, frame_shift)
    filtered_segments = kaldi_obj.segments[kaldi_obj.segments['rec'] == rec]
    speakers = np.unique(
            [kaldi_obj.utt2spk[seg['utt']] for seg
                in filtered_segments]).tolist()
    if n_speakers is None:
        n_speakers = len(speakers)
    T = np.zeros((Y.shape[0], n_speakers), dtype=np.int32)
    for seg in filtered_segments:
        speaker_index = speakers.index(kaldi_obj.utt2spk[seg['utt']])
        start_frame = np.rint(
                seg['st'] * rate / frame_shift).astype(int)
        end_frame = np.rint(
                seg['et'] * rate / frame_shift).astype(int)
        rel_start = rel_end = None
        if start <= start_frame and start_frame < end:
            rel_start = start_frame - start
        if start < end_frame and end_frame <= end:
            rel_end = end_frame - start
        if rel_start is not None or rel_end is not None:
            T[rel_start:rel_end, speaker_index] = 1
    return Y, T
