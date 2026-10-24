"""Real synthesis with the pinned models, from source.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The models (1.3 GB and 0.7 GB) download into $SUBTITLD_MODELS_DIR, or
~/.cache/subtitld/models, on the first run, checked against pins.json.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from chatterbox_addon import __main__ as addon, audio, download, dsp, engine, voices  # noqa: E402

pytestmark = pytest.mark.model


def _load(name):
    download.ensure(addon.pins(name), lambda *_: None, lambda: False)
    return engine.load(name, addon.model_files(name), engine.default_threads())


@pytest.fixture(scope='module')
def multilingual():
    return _load('multilingual')


@pytest.fixture(scope='module')
def turbo():
    return _load('turbo')


def _preset_reference(model, index=0):
    samples, rate = audio.read_wav(voices.clip_path(voices.presets()[index]))
    return model.prepare_reference(dsp.resample(samples, rate, engine.SAMPLE_RATE))


def _check_speech(wav, low=0.8, high=8):
    seconds = len(wav) / engine.SAMPLE_RATE
    assert low < seconds < high, seconds
    assert wav.dtype == np.float32 and np.isfinite(wav).all()
    assert 0.01 < float(np.sqrt(np.mean(wav ** 2))) < 0.5


def test_numpy_embeddings_match_the_graph(multilingual):
    """The prompt the add-on builds is what the export's embedding graph gives."""
    graph = engine._session(addon.model_files('multilingual')['embed_tokens'], 1)
    tokenizer = multilingual.tokenizer
    ids = tokenizer.encode('Musimy wyjść przed świtem.', 'pl')
    # The graph takes the template's sequence: exaggeration, text, two starts.
    sequence = np.asarray([[6563, *ids, engine.START_SPEECH, engine.START_SPEECH]], dtype=np.int64)
    positions = np.where(sequence >= engine.START_SPEECH, 0, np.arange(sequence.shape[1])[None] - 1)
    expected = graph.run(None, {'input_ids': sequence, 'position_ids': positions,
                                'exaggeration': np.asarray([multilingual.exaggeration], dtype=np.float32)})[0]
    reference = _preset_reference(multilingual)
    prompt = multilingual._prompt('Musimy wyjść przed świtem.', 'pl', reference, guidance=True)
    skip = reference.conditioning.shape[1]
    np.testing.assert_allclose(prompt[0, skip:], expected[0], atol=1e-6)
    # The unconditional row keeps the positions and drops the text.
    text_rows = slice(skip + 1, skip + 1 + len(ids))
    np.testing.assert_allclose(prompt[1, text_rows], multilingual.text_positions[:len(ids)], atol=1e-6)
    np.testing.assert_allclose(prompt[1, :text_rows.start], prompt[0, :text_rows.start])


def test_every_preset_prepares(multilingual):
    for index, voice in enumerate(voices.presets()):
        ref = _preset_reference(multilingual, index)
        assert ref.speaker.shape == (1, 192), voice.id
        assert ref.conditioning.shape[0] == 1 and ref.conditioning.shape[2] == 1024, voice.id
        assert ref.prompt_tokens.shape[1] > 100 and ref.features.shape[2] == 80, voice.id


@pytest.mark.parametrize('language,text', [
    ('en', 'We have to leave before sunrise.'),
    ('pt', 'Temos de sair antes do nascer do sol.'),
    ('pl', 'Musimy wyjść przed świtem.'),
    ('ko', '해가 뜨기 전에 떠나야 해요.'),
])
def test_multilingual_speaks(multilingual, language, text):
    _check_speech(multilingual.synthesize(text, _preset_reference(multilingual), language=language, seed=3))


def test_turbo_speaks_with_tags(turbo):
    _check_speech(turbo.synthesize('Oh, that is funny! [laugh] Anyway, see you tomorrow.',
                                   _preset_reference(turbo, 1), seed=3), high=10)


def test_a_short_clip_is_looped_to_six_seconds(turbo):
    samples, rate = audio.read_wav(voices.clip_path(voices.presets()[0]))
    short = dsp.resample(samples[:int(3 * rate)], rate, engine.SAMPLE_RATE)
    _check_speech(turbo.synthesize('We have to leave before sunrise.', turbo.prepare_reference(short), seed=1))


def test_the_same_line_comes_out_the_same(multilingual, monkeypatch):
    """The same speech tokens, so the same words, timing and voice; the
    decoder's own noise is not seeded, so not the same samples."""
    ref = _preset_reference(multilingual)
    tokens = []
    generate = multilingual._generate
    monkeypatch.setattr(multilingual, '_generate', lambda *a, **k: tokens.append(generate(*a, **k)) or tokens[-1])
    a = multilingual.synthesize('Same line, same voice.', ref, language='en', seed=11)
    b = multilingual.synthesize('Same line, same voice.', ref, language='en', seed=11)
    assert tokens[0] == tokens[1] and len(a) == len(b)
    level = lambda wav: float(np.sqrt(np.mean(wav ** 2)))  # noqa: E731
    assert abs(level(a) - level(b)) < 0.25 * level(a)


def test_cancel_stops_between_steps(multilingual):
    ref = _preset_reference(multilingual)
    calls = []
    flag = threading.Event()

    def progress(value):
        calls.append(value)
        flag.set()

    with pytest.raises(engine.Cancelled):
        multilingual.synthesize('A line that is cancelled as soon as it starts to be spoken.', ref,
                                language='en', progress=progress, cancelled=flag.is_set)
    assert calls


def test_a_stuck_line_is_tried_again(multilingual, monkeypatch):
    """The first attempt is made to repeat one token; the guided retry ends."""
    real = engine._penalize
    attempts = []

    def stuck_first_attempt(logits, tokens):
        logits = real(logits, tokens)
        if len(tokens) == 1:          # [start of speech]: an attempt begins
            attempts.append(1)
        if len(attempts) == 1 and len(tokens) > 5:
            logits[100] = 1e9
        return logits

    monkeypatch.setattr(engine, '_penalize', stuck_first_attempt)
    _check_speech(multilingual.synthesize('We have to leave before sunrise.', _preset_reference(multilingual),
                                          language='en', seed=2))
    assert len(attempts) == 2
