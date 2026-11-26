"""Convert an instrumented exact solve transcript to revmodel pvs-trace-check input."""

import argparse
import ast
import json
from pathlib import Path
from typing import TextIO


def require(condition, message):
    if not condition:
        raise ValueError(message)


def no_move(square):
    return None if square == 255 else square


def thread_number(worker):
    return int(worker.removeprefix("ThreadId(").removesuffix(")"))


def append_record(output, kind, record, frame=None):
    identifier = "" if frame is None else f"{frame}|"
    output.write(f"{kind}|{identifier}{json.dumps(record, separators=(',', ':'))}\n")


def normalize(source: TextIO, output: TextIO) -> None:
    frames = {}
    returned = {}
    nodes = {}
    passes = {}
    last_probe = {}
    selected_hit = {}
    cuts = {}
    pending_split = {}
    splits = {}
    table_scopes = []
    last_publication = {}

    def child(frame, last, worker, pv, parent_score):
        data = frames[last]
        require(
            (data["depth"], data["worker"], data["pv"], returned[last])
            == (frames[frame]["depth"] - 1, thread_number(worker), pv, -parent_score),
            "child return does not match the consuming search",
        )
        return data["parent"], dict(frame=last, pv=pv, alpha=data["alpha"], beta=data["beta"], score=-parent_score)

    def serial(phase, frame, worker, last, args):
        if phase == "tt_hit":
            slot, sequence, metadata = map(int, args)
            probe = last_probe[worker]
            require((slot, sequence, metadata) == (probe["slot"], probe["first"], probe["metadata"]),
                    "selected table hit disagrees with the sampled slot")
            selected_hit[frame] = dict(scope=probe["scope"], slot=slot, sequence=sequence)
            return
        if phase == "tt_cut":
            require(frame not in cuts, "duplicate table cutoff")
            cuts[frame] = selected_hit.pop(frame), int(args[0])
            return
        if phase == "pass":
            require(frame not in passes, "duplicate pass return")
            passes[frame] = last, int(args[0])
            return
        if phase == "pool":
            player, opponent, root = map(int, args[:3])
            generated = ast.literal_eval(args[3])
            data = frames[frame]
            require((player, opponent) == (data["player"], data["opponent"]), "move pool changed its board")
            nodes[frame] = dict(
                player=player, opponent=opponent, root=bool(root), pv=data["pv"],
                alpha=data["alpha"], beta=data["beta"], generated=generated, moves=[],
                calls=[], selections=[], handoff=None, publication=None,
            )
            return
        node = nodes[frame]
        if phase == "pre":
            square, score = map(int, args)
            parent, result = child(frame, last, worker, False, score)
            require(parent == frame, "presearch returned to a different frame")
            node["calls"].append(dict(square=square, **result))
            node["pre"] = square, score
        elif phase == "start":
            idx, _, _ = map(int, args[:3])
            node["moves"] = ast.literal_eval(args[3])
            require(idx == int("pre" in node), "serial start has the wrong presearch offset")
            if "pre" in node:
                square, score = node.pop("pre")
                node["selections"].append(dict(index=0, freshEnd=len(node["moves"]), square=square, score=score))
        elif phase == "precut":
            square, score = node.pop("pre")
            remaining = list(node["generated"])
            remaining.remove(square)
            node["moves"] = [square, *remaining]
            node["selections"].append(dict(index=0, freshEnd=len(node["moves"]), square=square, score=score))
        elif phase == "take":
            index, fresh_end, square = map(int, args[:3])
            node["take"] = index, fresh_end, square
        elif phase in ("scout", "wide"):
            square, score = map(int, args[:2])
            require(phase == "wide" or args[2] == "0", "exact scout reduced its depth")
            parent, result = child(frame, last, worker, phase == "wide", score)
            require(parent == frame and node["take"][2] == square, "child does not match its selected move")
            node["calls"].append(dict(square=square, **result))
        elif phase == "accept":
            square, score = map(int, args)
            index, fresh_end, selected = node.pop("take")
            require(square == selected, "serial result belongs to a different move")
            node["selections"].append(dict(index=index, freshEnd=fresh_end, square=square, score=score))
        elif phase == "postpone":
            index, fresh_end, square = map(int, args)
            node["selections"].append(dict(index=index, freshEnd=fresh_end, square=square, score=None))
        elif phase == "handoff":
            start, _, alpha, best = map(int, args[:4])
            node["handoff"] = dict(start=start, moves=ast.literal_eval(args[4]), alpha=alpha, best=best, reports=[])
            require(worker not in pending_split, "owner handoff is already pending")
            pending_split[worker] = frame
        elif phase == "split_move":
            require(node["split_result"][0] == no_move(int(args[0])), "split owner move disagrees with caller")
        elif phase == "split_score":
            require(node.pop("split_result")[1] == int(args[0]), "split owner score disagrees with caller")
        elif phase == "store_move":
            score, square = map(int, args)
            node["stored"] = score, no_move(square)
        elif phase == "cancel":
            node["cancelled"] = True

    def split(phase, activation, worker, ticket, a, b, c, last):
        if phase == "start":
            if c:
                frame = pending_split.pop(worker)
                splits[activation] = frame, worker, {}
                handoff = nodes[frame]["handoff"]
                require((ticket, a, b) == (handoff["start"], frames[frame]["depth"], handoff["best"]),
                        "split start disagrees with caller")
            return
        if activation not in splits:
            return
        frame, owner, tickets = splits[activation]
        node = nodes[frame]
        handoff = node["handoff"]
        if phase in {"scout_result", "wide_load", "wide_result", "lock", "check", "move_report", "report"}:
            require(tickets[ticket]["worker"] == thread_number(worker), "split ticket changed its worker")
        if phase == "move_seed":
            require((worker, ticket, b, c) == (owner, handoff["start"], handoff["alpha"], node["beta"]),
                    "split seed disagrees with caller")
            handoff["seed"] = no_move(a)
        elif phase == "scout_load":
            require(ticket not in tickets and b == node["beta"] and bool(c) == node["pv"], "invalid scout ticket")
            tickets[ticket] = dict(index=ticket - 1, worker=thread_number(worker), scoutAlpha=a, wide=None)
        elif phase == "scout_result":
            row = tickets[ticket]
            _, result = child(frame, last, worker, False, a)
            alpha = row["scoutAlpha"]
            require((result["alpha"], result["beta"]) == (-(alpha + 1), -alpha), "split scout used another window")
            row["scout"] = result["score"]
            row["scoutFrame"] = last
        elif phase == "wide_load":
            require(b == node["beta"], "split re-search changed beta")
            tickets[ticket]["wide"] = dict(alpha=a)
        elif phase == "wide_result":
            wide = tickets[ticket]["wide"]
            _, result = child(frame, last, worker, True, a)
            require((result["alpha"], result["beta"]) == (-node["beta"], -wide["alpha"]),
                    "split re-search used another window")
            wide["score"] = result["score"]
            wide["frame"] = last
        elif phase == "check":
            tickets[ticket]["stopped"] = bool(a)
        elif phase == "move_report":
            row = tickets[ticket]
            require(not row.pop("stopped"), "an interrupted worker reported a move")
            row.update(square=a, moveAfter=no_move(b), alphaAfter=c)
        elif phase == "report":
            row = tickets.pop(ticket)
            row.update(score=a, bestAfter=b)
            handoff["reports"].append(row)
        elif phase == "move_read":
            require(worker == owner, "split move was read by a different owner")
            node["split_result"] = no_move(a), b
        elif phase == "read":
            require(node["split_result"][1] == a, "split owner score disagrees with its move read")
            del splits[activation]

    for line, raw in enumerate(source, 1):
        fields = raw.rstrip("\r\n").split("|")
        try:
            if fields[0] == "trace":
                _, phase, frame, parent, worker, depth, score, sequence, kind, low, high, pv, player, opponent = fields
                frame = int(frame)
                data = dict(parent=int(parent), depth=int(depth), alpha=int(low), beta=int(high),
                            pv=bool(int(pv)), worker=thread_number(worker), player=int(player), opponent=int(opponent))
                if phase == "enter":
                    require(frame not in frames, "duplicate frame entry")
                    frames[frame] = data
                elif phase == "publish" and frame in nodes:
                    node = nodes[frame]
                    require(kind in {"ordinary", "presearch"} and node["publication"] is None,
                            "unexpected publication in a running serial frame")
                    stamp = last_publication.pop(worker)
                    require(stamp["sequence"] == int(sequence), "frame publishes a different table sequence")
                    require(node["stored"][0] == int(score), "frame published a different score")
                    node["publication"] = stamp
                elif phase == "return":
                    require(frames[frame] == data and frame not in returned, "return changed its frame")
                    returned[frame] = int(score)
                    append_record(output, "frame", dict(data, score=int(score)), frame)
                    selected_hit.pop(frame, None)
                    if frame in cuts:
                        sample, cutoff_score = cuts.pop(frame)
                        require(cutoff_score == int(score), "table cutoff changed its returned score")
                        append_record(output, "cut", sample, frame)
                    if frame in passes:
                        previous, score_at_pass = passes.pop(frame)
                        require(score_at_pass == int(score), "pass changed its returned score")
                        append_record(output, "pass", previous, frame)
                    node = nodes.pop(frame, None)
                    if node is not None and not node.get("cancelled") and "stored" in node:
                        saved, move = node.pop("stored")
                        require(saved == int(score), "saved score disagrees with function return")
                        append_record(output, "node", dict(node, result=saved, move=move), frame)
            elif fields[0] == "serial":
                _, phase, frame, worker, last, *args = fields
                serial(phase, int(frame), worker, int(last), args)
            elif fields[0] == "split":
                _, phase, activation, worker, ticket, a, b, c, last = fields
                split(phase, int(activation), worker, *map(int, (ticket, a, b, c, last)))
            elif fields[0] == "table_scope":
                _, base, size = fields
                base, size = int(base), int(size)
                require(size > 0, "empty table allocation")
                table_scopes.append((base, base + size, line))
            elif fields[0] == "table":
                _, kind, worker, *words = fields
                names = ("slot", "first", "last", "player", "opponent", "metadata")
                record = dict(zip(names, map(int, words), strict=True))
                slot = record["slot"]
                scope = next((start for lo, hi, start in reversed(table_scopes) if lo <= slot < hi), None)
                require(scope is not None, "table slot has no recorded initialization")
                thread = thread_number(worker)
                sample = dict(record, worker=thread, scope=scope)
                append_record(output, kind, sample)
                if kind == "probe":
                    last_probe[worker] = sample
                if kind == "publish":
                    last_publication[worker] = dict(scope=scope, slot=slot, sequence=record["first"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"line {line}: {raw.rstrip()}: {error}") from error
    require(not (nodes or passes or splits or pending_split or selected_hit or cuts),
            "trace ends with unfinished recorded work")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("certificate", type=Path)
    args = parser.parse_args()
    with args.trace.open(encoding="utf-8") as source, args.certificate.open("w", encoding="utf-8") as out:
        normalize(source, out)


if __name__ == "__main__":
    main()
