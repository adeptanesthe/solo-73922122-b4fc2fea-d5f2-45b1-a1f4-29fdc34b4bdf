"""Tests for pruning localisation (``state.diagnose`` and ``/api/diagnose``).

The diagnosis separates two reasons why a correct subtitle is missing from
the beam: the acoustics do not support it (``unreachable``), or beam search
cut the last viable path (``pruned``, with the exact frame and the candidates
kept there).  It must never change decoding progress.
"""

import math

import pytest
from fastapi.testclient import TestClient

from app.decoder import CTCBeamDecoder, NEG_INF, compute_prefix_alpha, prefix_alpha_scores
from app.main import app
from app.state import (
    MAX_DIAGNOSTIC_FRAMES,
    MAX_TARGET_CHARS,
    SessionError,
    SessionManager,
    advance,
    confirm,
    diagnose,
)

VOCAB = ["<blank>", "a", "b"]


def make_session(frames, width=1, vocab=VOCAB):
    manager = SessionManager()
    return manager, manager.create(vocab, width, frames)


def logp_of(frames, text, vocab=VOCAB):
    """Exact forward log probability of ``text`` over all frames."""
    import numpy as np

    with np.errstate(divide="ignore"):
        logf = np.log(np.asarray(frames, dtype=np.float64))
    ids = [vocab.index(ch) for ch in text]
    return prefix_alpha_scores(compute_prefix_alpha(logf, ids))[0]


# --------------------------------------------------------------------------- verdicts


def test_pruned_reports_first_frame_and_retained_candidates():
    # Width 1 keeps "a" after frame 1; the "b" path leaves the beam there.
    frames = [[0.4, 0.5, 0.1], [0.4, 0.5, 0.1]]
    _, session = make_session(frames, width=1)
    advance(session, 2)

    result = diagnose(session, "b")

    assert result["status"] == "pruned"
    assert result["prune_frame"] == 1
    assert [b["text"] for b in result["beams"]] == ["a"]
    # At the previous boundary the empty prefix could still complete "b".
    assert [b["text"] for b in result["last_completing"]] == [""]
    # Acoustically reachable: the exact forward probability is finite.
    assert result["target_logp"] is not None
    assert math.isclose(result["target_logp"], logp_of(frames, "b"), rel_tol=1e-12)
    assert result["frame"] == 2
    assert result["beam_width"] == 1


def test_pruned_retained_prefix_may_be_a_dead_end():
    # "a" stays in the beam and IS a prefix of "ab", yet after the last frame
    # it can no longer be completed — the diagnosis must not confuse "is a
    # prefix" with "can still reach the target".
    frames = [[0.4, 0.6, 0.0], [0.5, 0.0, 0.5]]
    _, session = make_session(frames, width=1)
    advance(session, 2)

    result = diagnose(session, "ab")

    assert result["status"] == "pruned"
    assert result["prune_frame"] == 2
    assert [b["text"] for b in result["beams"]] == ["a"]
    assert [b["text"] for b in result["last_completing"]] == ["a"]
    assert result["target_logp"] is not None


def test_unreachable_when_every_path_has_zero_probability():
    # "b" has probability 0 in every frame: combinatorially plausible, but
    # zero probability does not count as reachable.
    frames = [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]
    _, session = make_session(frames, width=4)
    advance(session, 2)

    result = diagnose(session, "ab")

    assert result["status"] == "unreachable"
    assert result["target_logp"] is None
    assert "prune_frame" not in result


def test_repeated_characters_need_an_intervening_blank():
    # Two "a" emissions with no blank in between collapse to "a": "aa" is
    # acoustically unreachable.
    frames = [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    _, session = make_session(frames, width=4)
    advance(session, 2)
    assert diagnose(session, "aa")["status"] == "unreachable"

    # With an explicit blank the same text becomes reachable and stays in beam.
    frames = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    _, session = make_session(frames, width=4)
    advance(session, 3)
    result = diagnose(session, "aa")
    assert result["status"] == "in_beam"
    assert result["beam_entry"]["text"] == "aa"


def test_in_beam_reports_rank_and_scores():
    frames = [[0.4, 0.5, 0.1]]
    _, session = make_session(frames, width=2)
    advance(session, 1)

    result = diagnose(session, "a")

    assert result["status"] == "in_beam"
    assert result["rank"] == 1
    assert result["beam_entry"]["text"] == "a"
    assert math.isclose(result["beam_entry"]["logp"], math.log(0.5), rel_tol=1e-12)


def test_prune_frame_matches_actual_beam_replay():
    # The reported frame must be exactly where a real decode loses the target:
    # completable at prune_frame - 1, gone at prune_frame, and the retained
    # candidates equal the decoder's own beam at that boundary.
    frames = [[0.4, 0.5, 0.1], [0.4, 0.5, 0.1], [0.3, 0.6, 0.1]]
    _, session = make_session(frames, width=1)
    advance(session, 3)

    result = diagnose(session, "b")
    assert result["status"] == "pruned"

    import numpy as np

    with np.errstate(divide="ignore"):
        logf = np.log(np.asarray(frames, dtype=np.float64))
    decoder = CTCBeamDecoder(VOCAB, 1)
    replayed = decoder.advance(decoder.initial_beams(), logf[: result["prune_frame"]])
    assert [b["text"] for b in result["beams"]] == [b.text for b in replayed]
    assert result["prune_frame"] == 1  # width 1 drops "b" at the first frame


def test_later_frames_cannot_hide_an_earlier_prune():
    # Once every completing path has left the beam it can never come back:
    # the first such frame is reported even though decoding continues.
    frames = [[0.4, 0.5, 0.1]] + [[0.3, 0.6, 0.1]] * 5
    _, session = make_session(frames, width=1)
    advance(session, 6)
    result = diagnose(session, "b")
    assert result["status"] == "pruned"
    assert result["prune_frame"] == 1


# --------------------------------------------------------------------------- read-only


@pytest.mark.parametrize(
    "frames,width,target",
    [
        ([[0.4, 0.5, 0.1], [0.4, 0.5, 0.1]], 1, "b"),      # pruned
        ([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]], 4, "ab"),     # unreachable
        ([[0.4, 0.5, 0.1]], 2, "a"),                       # in_beam
    ],
)
def test_diagnose_does_not_change_decoding_progress(frames, width, target):
    _, session = make_session(frames, width=width)
    advance(session, len(frames))
    before = (
        session.next_frame,
        session.revision,
        session.prefix,
        len(session.confirmations),
        [(b.text, b.logp, b.logp_blank, b.logp_nonblank) for b in session.beams],
    )
    diagnose(session, target)
    after = (
        session.next_frame,
        session.revision,
        session.prefix,
        len(session.confirmations),
        [(b.text, b.logp, b.logp_blank, b.logp_nonblank) for b in session.beams],
    )
    assert before == after


# --------------------------------------------------------------------------- validation


def test_diagnose_requires_unconfirmed_session_with_frames():
    frames = [[0.4, 0.5, 0.1], [0.4, 0.5, 0.1]]
    _, session = make_session(frames, width=2)
    with pytest.raises(SessionError, match="推进"):
        diagnose(session, "a")  # no frames consumed yet

    advance(session, 1)
    confirm(session, "a")
    with pytest.raises(SessionError, match="确认"):
        diagnose(session, "ab")


def test_diagnose_rejects_long_or_foreign_targets():
    frames = [[0.4, 0.5, 0.1]]
    _, session = make_session(frames, width=2)
    advance(session, 1)
    with pytest.raises(SessionError, match="不能为空"):
        diagnose(session, "")
    with pytest.raises(SessionError, match=f"{MAX_TARGET_CHARS}"):
        diagnose(session, "a" * (MAX_TARGET_CHARS + 1))
    with pytest.raises(SessionError, match="词表外"):
        diagnose(session, "ax")
    # blank is not a valid target character
    with pytest.raises(SessionError, match="词表外"):
        diagnose(session, "<blank>")


def test_diagnose_rejects_sessions_beyond_frame_limit():
    frames = [[0.4, 0.5, 0.1]] * (MAX_DIAGNOSTIC_FRAMES + 1)
    _, session = make_session(frames, width=2)
    advance(session, MAX_DIAGNOSTIC_FRAMES + 1)
    with pytest.raises(SessionError, match=f"{MAX_DIAGNOSTIC_FRAMES}"):
        diagnose(session, "a")
    # Exactly at the limit it still works.
    _, session = make_session(frames, width=2)
    advance(session, MAX_DIAGNOSTIC_FRAMES)
    assert diagnose(session, "a")["frame"] == MAX_DIAGNOSTIC_FRAMES


# --------------------------------------------------------------------------- HTTP


FRAMES = [[0.4, 0.5, 0.1], [0.4, 0.5, 0.1], [0.3, 0.6, 0.1]]


@pytest.fixture()
def client():
    return TestClient(app)


def create(client, frames=FRAMES, width=1):
    response = client.post(
        "/api/sessions",
        json={"vocab": VOCAB, "beam_width": width, "frames": frames},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_diagnose_endpoint_pruned(client):
    session = create(client)
    sid = session["session_id"]
    client.post("/api/advance", json={"session_id": sid, "to_frame": 3})
    before = client.get(f"/api/sessions/{sid}").json()

    response = client.post("/api/diagnose", json={"session_id": sid, "target": "b"})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "pruned"
    assert payload["prune_frame"] == 1
    assert payload["frame"] == 3
    assert payload["target"] == "b"
    assert payload["beam_width"] == 1
    assert payload["target_logp"] is not None
    assert [b["text"] for b in payload["beams"]] == ["a"]
    assert payload["last_completing"]

    # Read-only: the session state, revision included, is untouched.
    after = client.get(f"/api/sessions/{sid}").json()
    assert after == before


def test_diagnose_endpoint_unreachable_and_in_beam(client):
    session = create(client, width=4)
    sid = session["session_id"]
    client.post("/api/advance", json={"session_id": sid, "to_frame": 3})

    response = client.post("/api/diagnose", json={"session_id": sid, "target": "a"})
    # "a" is the strongest hypothesis over these frames: still in the beam.
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "in_beam"
    assert payload["rank"] == 1
    assert payload["beam_entry"]["text"] == "a"

    frames = [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]
    session = create(client, frames=frames, width=4)
    sid = session["session_id"]
    client.post("/api/advance", json={"session_id": sid, "to_frame": 2})
    response = client.post("/api/diagnose", json={"session_id": sid, "target": "ab"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "unreachable"
    assert payload["target_logp"] is None


def test_diagnose_endpoint_validation_errors(client):
    session = create(client)
    sid = session["session_id"]

    # Not advanced yet.
    assert client.post(
        "/api/diagnose", json={"session_id": sid, "target": "a"}
    ).status_code == 409

    client.post("/api/advance", json={"session_id": sid, "to_frame": 2})
    for target in ("", "a" * 17, "ax"):
        response = client.post("/api/diagnose", json={"session_id": sid, "target": target})
        assert response.status_code == 409, (target, response.text)

    # Confirmed sessions are rejected until the confirmation is undone.
    state = client.get(f"/api/sessions/{sid}").json()
    client.post("/api/confirm", json={"session_id": sid, "prefix": state["beams"][0]["text"]})
    assert client.post(
        "/api/diagnose", json={"session_id": sid, "target": "a"}
    ).status_code == 409
    client.post("/api/undo", json={"session_id": sid})
    assert client.post(
        "/api/diagnose", json={"session_id": sid, "target": "a"}
    ).status_code == 200

    # Unknown session.
    assert client.post(
        "/api/diagnose", json={"session_id": "nope", "target": "a"}
    ).status_code == 404


def test_diagnose_endpoint_beyond_frame_limit(client):
    frames = [[0.4, 0.5, 0.1]] * (MAX_DIAGNOSTIC_FRAMES + 1)
    session = create(client, frames=frames)
    sid = session["session_id"]
    client.post("/api/advance", json={"session_id": sid, "to_frame": MAX_DIAGNOSTIC_FRAMES + 1})
    response = client.post("/api/diagnose", json={"session_id": sid, "target": "a"})
    assert response.status_code == 409
    assert str(MAX_DIAGNOSTIC_FRAMES) in response.json()["detail"]


def test_diagnose_queued_behind_slow_advance_sees_consistent_state(monkeypatch):
    """A diagnose issued while a slow advance is running is applied after it
    on the session's single worker: it sees the post-advance state, and the
    advance's result is not disturbed."""
    import asyncio
    import time

    import httpx

    from app import state as state_module

    original = state_module.advance

    def slow_advance(session, to_frame):
        time.sleep(0.3)
        return original(session, to_frame)

    monkeypatch.setattr(state_module, "advance", slow_advance)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            created = (
                await client.post(
                    "/api/sessions",
                    json={"vocab": VOCAB, "beam_width": 1, "frames": [[0.4, 0.5, 0.1]] * 3},
                )
            ).json()
            sid = created["session_id"]
            await client.post("/api/advance", json={"session_id": sid, "to_frame": 1})
            advance_task = asyncio.create_task(
                client.post("/api/advance", json={"session_id": sid, "to_frame": 3})
            )
            await asyncio.sleep(0.05)
            diag_task = asyncio.create_task(
                client.post("/api/diagnose", json={"session_id": sid, "target": "b"})
            )
            advance_response, diag_response = await asyncio.gather(advance_task, diag_task)
            final = (await client.get(f"/api/sessions/{sid}")).json()
            return advance_response, diag_response, final

    advance_response, diag_response, final = asyncio.run(scenario())
    assert advance_response.status_code == 200
    assert diag_response.status_code == 200
    payload = diag_response.json()
    # Ran after the slow advance: it reports the post-advance frame count.
    assert payload["frame"] == 3
    assert payload["status"] == "pruned"
    assert payload["prune_frame"] == 1
    # Two advances happened; the read-only diagnose did not bump the revision.
    assert final["next_frame"] == 3
    assert final["revision"] == 2
