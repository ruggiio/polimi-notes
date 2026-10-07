#!/usr/bin/env python3
"""
board_selftest.py — Controlli deterministici del fail-safe della lavagna (niente video, niente modelli):
normalizzazione LaTeX, allineamento delle due letture, riconciliazione con letture e arbitro finti,
controllo a valle degli appunti.

  .venv/bin/python tests/board_selftest.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.board.board as B  # noqa: E402


def test_normalize():
    n = B.normalize
    # solo differenze di scrittura → uguali
    assert n(r"\dfrac{d\vec{a}}{dt} = L\,\vec{a}") == n(r"\frac{d}{dt}\vec a = L \vec a")
    assert n(r"\vec{\xi}_k = \underset{\vec{\xi}_k}{\arg\min} \|x\|") == n(r"\vec{\xi}_k=\arg\min_{\vec{\xi}_k}\|x\|")
    assert n(r"[\vec a_1 \mid \ldots \mid \vec a_m]") == n(r"[\vec{a}_1|\dots|\vec{a}_m]")
    assert n(r"\left( x \right)") == n("(x)")
    assert n(r"\widetilde U \widetilde\Sigma") == n(r"\tilde{U}\tilde{\Sigma}")
    # differenze di sostanza → diverse
    assert n(r"\Theta_k(\vec a)") != n(r"\theta_k(\vec a)")
    assert n(r"\tilde{\Sigma}^{-1}") != n(r"\Sigma^{-1}")
    assert n(r"\mathbb{V}\vec a") != n(r"V\vec a")
    assert n(r"\vec{u}") != n(r"u")
    assert n("A^T") != n("A")


def test_align():
    a = [r"X = U\Sigma Z^T", r"A = X' X^\dagger", r"\vec a(t_1)"]
    b = [r"\vec{a}(t_1)", r"A = X'X^{\dagger}", r"X = \tilde U \Sigma Z^T", r"\lambda_j"]
    pairs = B._align(a, b)
    assert (1, 1) in pairs and (2, 0) in pairs            # ordine diverso: abbinate lo stesso
    assert (0, 2) in pairs                                 # simili ma diverse: vanno all'arbitro
    assert (None, 3) in pairs                              # vista solo da B


def _fake_calls(responses: dict):
    """_cached_call finto: risponde in base al nome del file di cache (read_a / read_b / judge)."""
    def call(cache, prompt, system, model, timeout):
        for key, text in responses.items():
            if key in cache.name:
                return (text(prompt) if callable(text) else text), 0.0
        raise AssertionError(cache.name)
    return call


def _items(*formulas, text=None):
    items = ([{"kind": "text", "text": text}] if text else []) + [{"kind": "formula", "latex": f} for f in formulas]
    return json.dumps({"items": items})


def test_reconcile(tmp: Path):
    img = tmp / "board_01.png"
    img.write_bytes(b"")
    seen = {}

    def judge(prompt):
        seen["prompt"] = prompt
        # #1: Θ vs θ → A;  #2: Σ̃ vs Σ → neither (illeggibile);  #3: solo B → B
        return json.dumps([{"k": 1, "verdict": "A"}, {"k": 2, "verdict": "neither", "latex": None},
                           {"k": 3, "verdict": "B"}])

    orig = B._cached_call
    B._cached_call = _fake_calls({
        "read_a": _items(r"\dot{\vec a} = \Theta(\vec a)\vec\xi", r"A = X'X^\dagger",
                         r"X^\dagger = \tilde Z \tilde\Sigma^{-1} \tilde U^T", r"c_i [?]", text="SINDy"),
        "read_b": _items(r"\dot{\vec{a}} = \theta(\vec{a})\,\vec{\xi}", r"A = X' X^{\dagger}",
                         r"X^\dagger = \tilde Z \Sigma^{-1} \tilde U^T", r"c_i [?]", r"\lambda_j"),
        "judge": judge,
    })
    try:
        items, _ = B.read_shot(img, ("m1", "m2", "m3"), log=lambda *a: None)
    finally:
        B._cached_call = orig
    f = {B.normalize(x["latex"]): x for x in items if x.get("kind") == "formula"}
    st = {k: (v["status"], v.get("check")) for k, v in f.items()}
    assert st[B.normalize(r"A = X'X^\dagger")] == ("verified", "2/2")
    assert st[B.normalize(r"\dot{\vec a} = \Theta(\vec a)\vec\xi")] == ("verified", "2/3")
    assert st[B.normalize(r"X^\dagger = \tilde Z \tilde\Sigma^{-1} \tilde U^T")][0] == "unreadable"
    assert st[B.normalize(r"\lambda_j")] == ("verified", "2/3")
    # due lettori d'accordo su un simbolo illeggibile: mai "verificata"
    assert st[B.normalize(r"c_i [?]")][0] == "uncertain"
    assert "SINDy" in [x.get("text") for x in items]
    assert "(not transcribed)" in seen["prompt"]


def test_sheet(tmp: Path):
    img = tmp / "sheet_01.png"
    img.write_bytes(b"")
    seen = {}

    def judge(prompt):
        seen["prompt"] = prompt
        return json.dumps([{"k": 1, "verdict": "B"}])

    def items(*rows):
        return json.dumps({"items": [{"panel": p, "kind": "formula", "latex": f} for p, f in rows]})

    orig = B._cached_call
    B._cached_call = _fake_calls({
        "read_a": items((1, r"A = X'X^\dagger"), (2, r"\dot{\vec a} = u(\vec a)\vec\xi")),
        "read_b": items((1, r"A = X' X^{\dagger}"), (2, r"\dot{\vec a} = \Theta(\vec a)\vec\xi")),
        "judge": judge,
    })
    try:
        res, _ = B.read_sheet(img, 2, ("m1", "m2", "m3"), log=lambda *a: None)
    finally:
        B._cached_call = orig
    assert res[1][0]["status"] == "verified" and res[1][0]["check"] == "2/2"
    assert res[2][0]["status"] == "verified" and "Theta" in res[2][0]["latex"]
    assert "(panel 2)" in seen["prompt"] and "(panel 1)" not in seen["prompt"]


def test_json():
    # LaTeX con backslash non raddoppiati nelle risposte dei modelli
    assert B._json('[{"latex": "\\vec{a}"}]')[0]["latex"] == r"\vec{a}"            # JSON non valido: riparato
    assert B._json('[{"latex": "\\frac{a}{b}"}]')[0]["latex"] == r"\frac{a}{b}"      # valido ma \f = form feed
    assert B._json('[{"latex": "\\\\beta"}]')[0]["latex"] == r"\beta"             # già corretto: invariato
    assert B._json('{"text": "a\\nb"}')["text"] == "a\nb"                         # a capo legittimo
    assert B._json('[{"k": 1, latex: }]') is None


def test_fragments():
    full = {"kind": "formula", "status": "verified",
            "latex": r"\mathcal{L}_2(\vec w) = \sum_{i=1}^{n_s} \| \Phi(\vec\mu^{(i)}) - \hat\varphi(\vec u_h(\vec\mu^{(i)})) \|^2"}
    cut = {"kind": "formula", "status": "uncertain", "latex": r"[?] - \hat{\varphi}(\vec{u}_h(\vec{\mu}^{(i)}))"}
    tiny = {"kind": "formula", "status": "unreadable", "latex": r"\Phi)"}
    other = {"kind": "formula", "status": "uncertain", "latex": r"X^\dagger = \tilde Z \Sigma^{-1} [?]"}
    shots = [B.Shot(0, 30, [0], items=[full]), B.Shot(30, 60, [1], items=[cut, tiny, other])]
    assert B.resolve_fragments(shots) == 2
    assert cut["status"] == "covered" and tiny["status"] == "covered" and other["status"] == "uncertain"


def test_camera():
    import numpy as np
    rng = np.random.default_rng(0)
    flat = np.full((100, 300, 3), 40, np.uint8)                       # editor scuro: fondo piatto
    flat[::7, ::5] = 200                                              # testo
    board = np.clip(70 + rng.normal(0, 2.5, (100, 300, 1)), 0, 255).astype(np.uint8).repeat(3, -1)
    assert not B._camera(flat, B._board_mask(flat))
    assert B._camera(board, B._board_mask(board))


def test_check_notes():
    shot = B.Shot(t0=0, t1=30, strips=[0], image="x.png", items=[
        {"kind": "formula", "latex": r"A = X' X^{\dagger}", "status": "verified"},
        {"kind": "formula", "latex": r"\dot{\vec a} = \Theta(\vec a)\,\vec\xi", "status": "verified"},
        {"kind": "formula", "latex": r"X = \tilde U \tilde \Sigma \tilde Z^T", "status": "verified"},
        {"kind": "formula", "latex": r"\lambda_j", "status": "verified"},          # troppo corta: ignorata
    ])
    tex = r"""
Si ottiene \begin{equation} A = X'X^\dagger \label{eq:a} \end{equation}
e il modello \[ \dot{\vec{a}} = \theta(\vec{a}) \vec{\xi} \]
"""
    rep = {r["board"]: r["ratio"] for r in B.check_notes(tex, B.BoardReading([], [shot]))}
    assert rep[r"A = X' X^{\dagger}"] == 1.0
    assert rep[r"\dot{\vec a} = \Theta(\vec a)\,\vec\xi"] < 1.0        # Θ → θ: cambiata
    assert rep[r"X = \tilde U \tilde \Sigma \tilde Z^T"] < 0.85          # assente
    assert r"\lambda_j" not in rep


if __name__ == "__main__":
    import tempfile
    test_normalize()
    test_align()
    with tempfile.TemporaryDirectory() as d:
        test_reconcile(Path(d))
        test_sheet(Path(d))
    test_json()
    test_fragments()
    test_camera()
    test_check_notes()
    print("board_selftest: ok")
