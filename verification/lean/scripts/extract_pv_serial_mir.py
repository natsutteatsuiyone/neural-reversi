"""Check that internal PV owner transitions share Root numerical decisions and record PV writes."""

from itertools import product
import re

from mir_common import checked, blocks
from extract_header import CONDITIONS, initial, lmr_returns_zero, ordinary, specialization, transition
from extract_nonpv_leaf_mir import impl_constant
import extract_pv_owner_flow
from extract_root_update import decode, post_step
from extract_root_wipeout import FILENAME, constants


def verify(dump, repo):
    extract_pv_owner_flow.verify(dump, repo)
    claim = (dump / "reversi_core.search-pvs-claim_child.SimplifyCfg-initial.after.mir").read_text(encoding="utf-8")
    checked(claim.startswith("// MIR for `search::pvs::claim_child` after SimplifyCfg-initial\n")
            and claim.splitlines().count("fn claim_child(_1: &threading::thread::Thread, _2: &Board, _3: u32) -> Claim<'_> {") == 1,
            "unexpected claim_child signature")
    claim_body = {k: ordinary(code) for k, code in blocks(claim).items()}
    checked(claim_body == {
        0: ["_4 = const <SS as search::strategy::SearchStrategy>::IS_ENDGAME;", "switchInt(move _4) -> [0: bb5, otherwise: bb1];"],
        1: ["_6 = copy _3;", "_5 = Ge(move _6, const <SS as search::strategy::SearchStrategy>::MIN_CLAIM_DEPTH);",
            "switchInt(move _5) -> [0: bb4, otherwise: bb2];"],
        2: ["_7 = &(*_1);", "_8 = &(*_2);", "_0 = threading::thread::Thread::claim(move _7, move _8) -> [return: bb3, unwind: bb7];"],
        3: ["goto -> bb6;"], 4: ["goto -> bb5;"], 5: ["_0 = Claim::<'_>::Untracked;", "goto -> bb6;"], 6: ["return;"]},
        "unexpected claim_child dispatch or forwarded result")
    limit = impl_constant(repo, dump, "strategy", "impl SearchStrategy for EndGameStrategy {", "MIN_CLAIM_DEPTH", "u32")
    minimum = re.fullmatch(r"(\d+)_u32", limit)
    checked(minimum, "missing compiled claim threshold")
    checked(int(minimum[1]) > 5, f"MIN_CLAIM_DEPTH is {minimum[1]}, but smallClaim "
            "(PvsParallel/PvSerialMir.lean) needs shallow-search depths (5 and below) never to claim")
    kernel = dump / "reversi_core.search-pvs-compute_lmr_reduction.SimplifyCfg-initial.after.mir"
    lmr_returns_zero(kernel.read_text(encoding="utf-8"))
    specialization(repo)
    root = constants(repo)
    original = (dump / FILENAME).read_text(encoding="utf-8")
    cfg = blocks(original)
    regs, header = initial(original, cfg)
    internal = dict(root)
    for name, expected in (("PV_NODE", "true"), ("ROOT_NODE", "false")):
        bit = impl_constant(repo, dump, "node_type", "impl NodeType for PV {", name, "bool")
        checked(bit == expected, "unsupported internal PV specialization")
        internal[f"const <NT as search::node_type::NodeType>::{name}"] = int(bit == "true")
    for flags in product((False, True), repeat=len(CONDITIONS)):
        for tag in ([1] if flags[5] else [0, 2]):
            common = transition(cfg, regs, header, flags, root, tag)
            for cut in (0, 1):
                actual = transition(cfg, regs, header, flags, internal, tag, "pvID", cut)
                checked(actual == common, f"root and internal PV action differ: {flags}, {tag}, {cut}")
    cfg, regs, pc = decode(dump)
    for flags in product((False, True), repeat=5):
        reference = post_step(cfg, regs, root, pc, flags)
        actual = post_step(cfg, regs, internal, pc, flags, internal=True)
        checked(actual[1:] == reference[1:] and actual[0] == reference[3],
                f"internal PV numeric state or PV write differs: {flags}: {actual}")
    return f"""module
public import NeuralReversi.PvsParallel.RootHeaderMir
public import NeuralReversi.PvsParallel.RootUpdateMir

@[expose] public section

namespace NeuralReversi.PvsParallel.PvSerialMir

def minimumClaimDepth : Nat := {int(minimum[1])}

abbrev action := headerAction
abbrev decision := RootUpdateMir.decision
def writesPv (d : RootUpdateMir.Changes) : Bool := d.takesMove

end NeuralReversi.PvsParallel.PvSerialMir
"""
