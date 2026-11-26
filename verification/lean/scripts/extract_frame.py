"""Extract entry and return around the arbitrary-length Root / EndGameStrategy serial move loop."""

from dataclasses import dataclass, field
import itertools

from extract_header import ENDGAME, initial, binding
from extract_single import run, SIGNATURE, RESET
from mir_common import blocks, checked
from extract_root_wipeout import constants
from check_keys import verify_evaluators, verify_reordering
from check_serial_move_list_mir import check as check_serial_move_list

EVALUATE = "ordering::<impl MoveList>::evaluate_moves::<NT, SS>"
SORT = "ordering::<impl MoveList>::sort"
FILTER = "move_list::ordering|move_list::move_array|::iter_mut|postpone_move"


@dataclass
class Frame:
    start: int
    end: int | None
    inputs: dict = field(default_factory=dict)
    externals: dict = field(default_factory=dict)
    observe: tuple = ()


def render(mir, repo):
    check_serial_move_list(mir.parent)
    verify_evaluators(mir.parent)
    verify_reordering(mir.parent)
    source = mir.read_text(encoding="utf-8")
    checked(SIGNATURE in source.splitlines(), "unexpected PVS signature")
    cfg = blocks(source)
    names, header = initial(source, cfg)
    traits = constants(repo)
    traits[ENDGAME] = 1
    ends = [n for n, body in cfg.items() if " = transposition_table::Bound::classify::<NT>(" in body[-1]]
    checked(len(ends) == 1, "ambiguous root return epilogue")
    classify, = ends
    probe = binding(source, "tt_probe_result")
    raw = binding(source, "raw_eval")
    for multiple, hit in itertools.product((False, True), repeat=2):
        count = "multiple" if multiple else 1
        frame = Frame(0, header, externals={
            "MoveList::count": (["list"], count),
            "MoveList::wipeout_move": (["list"], ("some", "sq") if hit else "none"),
            EVALUATE: (["list", "ctx", "board", "depth", "hint", "alpha", 0], "unit"),
            SORT: (["list"], "unit"),
        }, observe=(EVALUATE, SORT))
        events, final = run(cfg, (0, 0, 0, 0, 0, 0), traits, frame)
        if hit:
            checked(events == [RESET, ("write", "sq", 16384, 1)] and final == 16384, "wrong wipeout return")
            continue
        checked(events == [RESET] + ([EVALUATE, SORT] if multiple else []), f"unexpected root entry effects {events}")
        expected = {"ctx": ("ref", "ctx"), "board": ("ref", "board"), "depth": "depth",
                    "alpha": "alpha", "org_alpha": "alpha", "beta": "beta", "thread": ("ref", "arc"),
                    "cut_node": 0, "all_node": 0, "tt_pv": 1, "move_list": "list",
                    "n_moves": count, "idx": 0, "fresh_end": count, "best_score": -32767, "best_move": "none"}
        for key, value in expected.items():
            checked(final[names[key]] == value, f"unexpected header binding {key}")
        checked(final[probe] == "probe" and final[raw] == "none", "unexpected probe/eval for root epilogue")
    for first_line, has_best in itertools.product((False, True), repeat=2):
        best_move = ("some", "sq") if has_best else "none"
        frame = Frame(classify, None, {
            names["alpha"]: "raisedAlpha", names["org_alpha"]: "alpha", names["best_score"]: "finalBest",
            names["best_move"]: best_move, names["tt_pv"]: 1, probe: "probe", raw: "none",
        })
        events, result = run(cfg, (0, 0, 0, 0, 0, first_line), traits, frame)
        stored = ("store", "tt", "slot", "board", result, ("bound", result), "depth",
                  best_move, "selectivity", 1, 1)
        checked(result == "finalBest" and events == ([stored] if first_line else []),
                "root epilogue published or returned wrong score")
    return """namespace NeuralReversi.PvsParallel

inductive RootEntry where
  | wipeout
  | header (sort : Bool) (idx fresh : Nat) (best : Int)
deriving DecidableEq, Repr

def rootEntry (count : Nat) (cacheHit : Bool) (_ : 0 < count) : RootEntry :=
  if cacheHit then .wipeout else .header (decide (1 < count)) 0 count (-32767)

def rootEpilogue (pvIdx : Nat) (best : Int) : Option Int × Int :=
  (if pvIdx = 0 then some best else none, best)

end NeuralReversi.PvsParallel
"""

