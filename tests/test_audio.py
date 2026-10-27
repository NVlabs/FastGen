# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from fastgen.utils.audio import resample_audio


@pytest.mark.parametrize(
    ("alias", "method"),
    [("sinc_interpolation", "sinc_interp_hann"), ("kaiser_window", "sinc_interp_kaiser")],
)
def test_deprecated_resampling_names_preserve_the_filter(alias, method):
    impulse = torch.zeros(1, 128)
    impulse[0, 64] = 1
    expected = resample_audio(impulse, 48000, 16000, resampling_method=method)

    with pytest.warns(UserWarning, match="deprecated"):
        actual = resample_audio(impulse, 48000, 16000, resampling_method=alias)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    ("length", "source_rate", "target_rate", "expected_length"),
    [(120427, 44100, 48000, 131078), (147, 44100, 48000, 160), (481, 48000, 16000, 161)],
)
def test_resampling_keeps_the_final_fractional_sample(dtype, length, source_rate, target_rate, expected_length):
    waveform = torch.ones(2, length, dtype=dtype)
    actual = resample_audio(waveform, source_rate, target_rate)

    assert actual.shape == (2, expected_length)
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[:, -1]) == 2
