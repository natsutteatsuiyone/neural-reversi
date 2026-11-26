------------------------- MODULE ConcurrentMoveList -------------------------
EXTENDS Integers, FiniteSets, Sequences, TLC

(*
 * Formal model of two interacting lock-free move-dispatch protocols in
 * reversi-core:
 *
 * 1. ConcurrentMoveIterator (`crates/reversi-core/src/move_list/iterator.rs`):
 *    Lock-free `AtomicUsize::try_update` weak-CAS loop distributing split-point
 *    moves across worker threads, with stale reads and bounded spurious weak-CAS
 *    failures.
 *
 * 2. InFlightTable + PvsPostpone
 *    (`crates/reversi-core/src/search/threading/in_flight.rs`,
 *     `crates/reversi-core/src/search/pvs.rs`,
 *     `crates/reversi-core/src/move_list.rs`):
 *    Each split-point worker claims its outer child hash in the direct-mapped
 *    `InFlightTable` (`search_split_point`), then executes a serial child move
 *    loop (`search`) that claims grandchild hashes, postpones `Claim::Busy`
 *    moves at most once when `!eldest && has_fresh_sibling` via
 *    `rotate_left(1)`, searches `Claim::Untracked` hash collisions immediately
 *    without disturbing the slot owner, and releases only `Claim::Owned` guards
 *    on drop.
 *)

CONSTANTS
    N,
    NumIterMoves,
    SpuriousBudget,
    Mutant

Workers == 1..N
IterMoves == 1..NumIterMoves

EMPTY == 0
Hashes == 1..4
Slots == 1..2
NO_CLAIM == <<"None", 0, 0>>
ClaimVal == {NO_CLAIM} \cup ({"Owned", "Busy", "Untracked"} \X Slots \X Hashes)

\* Direct-mapped slot index mirroring `hash as usize & (SLOTS - 1)`:
\* Hashes 1 and 3 collide in slot 1; hashes 2 and 4 collide in slot 2.
SlotOf(h) == ((h - 1) % 2) + 1

\* Child hash for each move yielded by ConcurrentMoveIterator.
IterHash(m) == ((m - 1) % 4) + 1

\* Serial MoveList (length 3) searched inside each split-point child:
\* - Move 1 (outer hash 1, slot 1) has non-eldest hash 2 (Busy against Move 2)
\*   and hashes 4, 3 (colliding Untracked against slots 2 and 1).
\* - Move 2 (outer hash 2, slot 2) has non-eldest hash 1 (Busy against Move 1)
\*   and hashes 3, 4 (colliding Untracked against slots 1 and 2).
\* - Move 3 (outer hash 3, slot 1, colliding with Move 1) has eldest hash 2
\*   (Busy against Move 2 on eldest, which must NOT postpone) plus 1, 4.
ChildMoves(m) ==
    CASE m = 1 -> <<4, 2, 3>>
      [] m = 2 -> <<3, 1, 4>>
      [] m = 3 -> <<2, 1, 4>>
      [] OTHER -> <<1, 2, 3>>

ChildMoveSet(m) == {ChildMoves(m)[i] : i \in 1..Len(ChildMoves(m))}

\* Mirrors `self.moves.as_mut_slice()[index..].rotate_left(1)` in move_list.rs.
RotateLeftFrom(seq, i) ==
    [j \in 1..Len(seq) |->
        IF j < i THEN seq[j]
        ELSE IF j < Len(seq) THEN seq[j + 1]
        ELSE seq[i]]

(* --algorithm ConcurrentMoveList

variables
    \* Shared ConcurrentMoveIterator state
    iterIdx = 1,
    spuriousLeft = SpuriousBudget,

    \* Shared InFlightTable slots (EMPTY or active hash key)
    inFlight = [s \in Slots |-> EMPTY],

    \* Active Claim guards per worker (outer = search_split_point, inner = search)
    outerClaim = [w \in Workers |-> NO_CLAIM],
    innerClaim = [w \in Workers |-> NO_CLAIM],

    \* Ghost tracking for safety invariants
    iterClaims = [m \in IterMoves |-> 0],
    iterDone = [m \in IterMoves |-> FALSE],
    innerSearched = [m \in IterMoves |-> {}],
    sawDrained = [w \in Workers |-> FALSE];

define
    OwnersOf(s) ==
        ({<<w, "outer">> : w \in {u \in Workers : outerClaim[u][1] = "Owned" /\ outerClaim[u][2] = s}})
        \cup
        ({<<w, "inner">> : w \in {u \in Workers : innerClaim[u][1] = "Owned" /\ innerClaim[u][2] = s}})

    AllWorkersDone == \A w \in Workers : pc[w] = "Done"

    NoDuplicateIteratorClaim ==
        \A m \in IterMoves : iterClaims[m] <= 1

    DrainedMeansAllClaimed ==
        (\E w \in Workers : sawDrained[w]) =>
            /\ iterIdx = NumIterMoves + 1
            /\ \A m \in IterMoves : iterClaims[m] >= 1

    ExclusiveInFlightOwnership ==
        \A s \in Slots : Cardinality(OwnersOf(s)) <= 1

    CollisionPreservesOwner ==
        /\ \A w \in Workers :
            /\ (outerClaim[w][1] = "Owned" => inFlight[outerClaim[w][2]] = outerClaim[w][3])
            /\ (innerClaim[w][1] = "Owned" => inFlight[innerClaim[w][2]] = innerClaim[w][3])
        /\ \A s \in Slots : (OwnersOf(s) = {}) => (inFlight[s] = EMPTY)

    NoDroppedMove ==
        /\ \A m \in IterMoves : iterDone[m] => innerSearched[m] = ChildMoveSet(m)
        /\ AllWorkersDone =>
            /\ \A m \in IterMoves : iterClaims[m] = 1 /\ iterDone[m]
            /\ \A s \in Slots : inFlight[s] = EMPTY

    EventuallyAllDone == <>AllWorkersDone
end define;

fair process Worker \in Workers
variables
    obsIdx = 0,
    curMove = 0,
    ml = <<>>,
    idx = 0,
    freshEnd = 0;
begin
IterLoad:
    obsIdx := iterIdx;

IterCAS:
    if obsIdx > NumIterMoves then
        sawDrained[self] := TRUE;
        obsIdx := 0;
        goto Done;
    elsif Mutant = "non_atomic_iterator_claim" then
        \* Bug: non-atomic load-then-store instead of CAS.
        iterIdx := obsIdx + 1;
        curMove := obsIdx;
        iterClaims[obsIdx] := iterClaims[obsIdx] + 1;
        obsIdx := 0;
    elsif iterIdx = obsIdx then
        either
            \* Weak CAS succeeds.
            iterIdx := obsIdx + 1;
            curMove := obsIdx;
            iterClaims[obsIdx] := iterClaims[obsIdx] + 1;
            obsIdx := 0;
        or
            \* Spurious weak-CAS failure (bounded budget).
            await spuriousLeft > 0;
            spuriousLeft := spuriousLeft - 1;
            obsIdx := iterIdx;
            goto IterCAS;
        end either;
    else
        \* Stale read: another worker advanced iterIdx.
        if Mutant = "stale_claim_skips_move" then
            \* Bug: advances cursor on stale retry instead of reloading.
            iterIdx := IF iterIdx <= NumIterMoves THEN iterIdx + 1 ELSE iterIdx;
        end if;
        obsIdx := iterIdx;
        goto IterCAS;
    end if;

OuterClaim:
    \* `search_split_point` claims the child position in InFlightTable.
    with h = IterHash(curMove), s = SlotOf(h) do
        if inFlight[s] = EMPTY then
            inFlight[s] := h;
            outerClaim[self] := <<"Owned", s, h>>;
        elsif inFlight[s] = h then
            if Mutant = "duplicate_owned_on_busy" then
                outerClaim[self] := <<"Owned", s, h>>;
            else
                outerClaim[self] := <<"Busy", s, h>>;
            end if;
        else
            outerClaim[self] := <<"Untracked", s, h>>;
        end if;
    end with;
    ml := ChildMoves(curMove);
    idx := 1;
    freshEnd := Len(ChildMoves(curMove));

InnerClaim:
    if idx > (IF Mutant = "drop_postponed_move" THEN freshEnd ELSE Len(ml)) then
        goto OuterDrop;
    else
        with h = ml[idx], s = SlotOf(h) do
            if inFlight[s] = EMPTY then
                inFlight[s] := h;
                innerClaim[self] := <<"Owned", s, h>>;
                goto InnerDrop;
            elsif inFlight[s] = h then
                if Mutant = "duplicate_owned_on_busy" then
                    innerClaim[self] := <<"Owned", s, h>>;
                    goto InnerDrop;
                elsif idx > 1 /\ (IF Mutant = "unbounded_postpone_livelock" THEN TRUE ELSE idx < freshEnd) then
                    \* Claim::Busy && !eldest && has_fresh_sibling -> postpone.
                    goto InnerPostpone;
                else
                    innerClaim[self] := <<"Busy", s, h>>;
                    goto InnerDrop;
                end if;
            else
                \* Different hash collided in the same slot: Claim::Untracked.
                innerClaim[self] := <<"Untracked", s, h>>;
                goto InnerDrop;
            end if;
        end with;
    end if;

InnerPostpone:
    ml := RotateLeftFrom(ml, idx);
    if Mutant # "unbounded_postpone_livelock" then
        freshEnd := freshEnd - 1;
    end if;
    goto InnerClaim;

InnerDrop:
    \* Finish searching ml[idx] and drop its Claim guard.
    innerSearched[curMove] := innerSearched[curMove] \cup {ml[idx]};
    if innerClaim[self][1] = "Owned"
       \/ (Mutant = "collision_clears_slot" /\ innerClaim[self][1] = "Untracked") then
        inFlight[innerClaim[self][2]] := EMPTY;
    end if;
    innerClaim[self] := NO_CLAIM;
    idx := idx + 1;
    goto InnerClaim;

OuterDrop:
    \* Finish child search and drop outer Claim guard from search_split_point.
    iterDone[curMove] := TRUE;
    if outerClaim[self][1] = "Owned"
       \/ (Mutant = "collision_clears_slot" /\ outerClaim[self][1] = "Untracked") then
        inFlight[outerClaim[self][2]] := EMPTY;
    end if;
    outerClaim[self] := NO_CLAIM;
    curMove := 0;
    ml := <<>>;
    idx := 0;
    freshEnd := 0;
    goto IterLoad;

end process;

end algorithm; *)
\* BEGIN TRANSLATION
\* END TRANSLATION

TypeOK ==
    /\ iterIdx \in 1..(NumIterMoves + 1)
    /\ spuriousLeft \in 0..SpuriousBudget
    /\ inFlight \in [Slots -> ({EMPTY} \cup Hashes)]
    /\ outerClaim \in [Workers -> ClaimVal]
    /\ innerClaim \in [Workers -> ClaimVal]
    /\ iterClaims \in [IterMoves -> 0..N]
    /\ iterDone \in [IterMoves -> BOOLEAN]
    /\ innerSearched \in [IterMoves -> SUBSET Hashes]
    /\ sawDrained \in [Workers -> BOOLEAN]
    /\ \A w \in Workers :
        /\ obsIdx[w] \in 0..(NumIterMoves + 1)
        /\ curMove[w] \in 0..NumIterMoves
        /\ Len(ml[w]) \in {0, 3}
        /\ \A i \in 1..Len(ml[w]) : ml[w][i] \in Hashes
        /\ idx[w] \in 0..4
        /\ freshEnd[w] \in 0..3
=============================================================================
