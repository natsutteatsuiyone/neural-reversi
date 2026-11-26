------------------------ MODULE TranspositionTable ------------------------
EXTENDS Integers, FiniteSets, Sequences, TLC

\* Bounded PlusCal/TLA+ model of the TranspositionTable SeqLock protocol
\* in crates/reversi-core/src/transposition_table.rs:
\*   - 1 cluster of CLUSTER_SIZE = 2 slots (Slots = {0, 1})
\*   - Lookup (read_for_lookup): scans slots, checks even seq1, board match,
\*     reads meta in a separate atomic step, validates seq1 = seq2 and
\*     bound # "None"
\*   - Probe + Save (probe, TTEntry::save, save_exact, try_load_snapshot,
\*     seqlock_write): selects slot via cluster scan, runs Exact bypass or
\*     non-Exact snapshot + replacement policy with bounded CAS retry loop
\*     (SAVE_RETRIES = 4)
\*   - Shared clear (clear_shared): clears each slot via SeqLock with
\*     fast-skip on empty entries and bounded CAS retry loop (SAVE_RETRIES = 4)

CONSTANTS
    NumWriters,
    NumReaders,
    MaxOps,
    MaxReaderOps,
    MaxRetries,
    Mutant

Slots == {0, 1}
Boards == {0, 1, 2}
Bounds == {"None", "Exact", "Lower", "Upper"}
Depths == {0, 3}
Scores == {0, 1, 2}
Gens == {0, 1}
AgeWeight == 8

Writers == 1..NumWriters
Readers == (NumWriters + 1)..(NumWriters + NumReaders)

EmptyMeta == [bound |-> "None", depth |-> 0, score |-> 0, gen |-> 0]
MetaType == [bound: Bounds, depth: Depths, score: Scores, gen: Gens]

NoSnap == [valid |-> FALSE, slot |-> 0, seq |-> 0, board |-> 0, meta |-> EmptyMeta]
SnapType == [valid: BOOLEAN, slot: Slots, seq: Nat, board: Boards, meta: MetaType]

NoReq == [kind |-> "None", board |-> 0, meta |-> EmptyMeta]

\* Compact representative operation catalogs:
\* - Writer 1 exercises strong vs shallow non-Exact same-board saves (both
\*   accepting and rejecting replacement via ShouldReplace).
\* - Writer 2+ exercises Exact bypass, non-Exact Upper cross-board/newer-gen
\*   overwrite, and clear_shared.
W1Ops == {
    [kind |-> "Save", board |-> 1,
     meta |-> [bound |-> "Lower", depth |-> 3, score |-> 1, gen |-> 0]],
    [kind |-> "Save", board |-> 1,
     meta |-> [bound |-> "Lower", depth |-> 0, score |-> 1, gen |-> 0]]
}

W2Ops == {
    [kind |-> "Save", board |-> 2,
     meta |-> [bound |-> "Exact", depth |-> 0, score |-> 2, gen |-> 1]],
    [kind |-> "Save", board |-> 2,
     meta |-> [bound |-> "Upper", depth |-> 3, score |-> 2, gen |-> 1]],
    [kind |-> "Clear", board |-> 0, meta |-> EmptyMeta]
}

OpsFor(w) == IF w = 1 THEN W1Ops ELSE W2Ops

RelativeAge(entryGen, curGen) == IF curGen = entryGen THEN 0 ELSE 1
ReplacementScore(m, curGen) == m.depth - RelativeAge(m.gen, curGen) * AgeWeight

ShouldReplace(newBoard, newMeta, storedBoard, storedMeta) ==
    \/ storedMeta.bound = "None"
    \/ newBoard # storedBoard
    \/ newMeta.depth + 2 >= storedMeta.depth
    \/ newMeta.gen # storedMeta.gen

\* Mirrors TranspositionTable::probe cluster victim selection over {0, 1}:
\* prefer empty or matching board; if both occupied & stable, pick lower
\* ReplacementScore (ties go to slot 0).
PickProbeSlot(b, g, sq, bd, mt) ==
    IF sq[0] % 2 = 0 /\ (mt[0].bound = "None" \/ bd[0] = b) THEN 0
    ELSE IF sq[1] % 2 = 0 /\ (mt[1].bound = "None" \/ bd[1] = b) THEN 1
    ELSE IF sq[0] % 2 = 0 /\ sq[1] % 2 = 0 THEN
        IF ReplacementScore(mt[1], g) < ReplacementScore(mt[0], g) THEN 1 ELSE 0
    ELSE IF sq[1] % 2 = 0 THEN 1
    ELSE 0

(* --algorithm TranspositionTable

variables
    \* Per-slot SeqLock state (CLUSTER_SIZE = 2)
    seq = [sl \in Slots |-> 0],
    board = [sl \in Slots |-> 0],
    meta = [sl \in Slots |-> EmptyMeta],

    \* Ghost history: published[sl][evenSeq] = <<board, meta>>
    published = [sl \in Slots |-> (0 :> <<0, EmptyMeta>>)],

    \* Ghost ownership set of writers/clearers currently inside the payload
    \* write critical section for each slot
    writeOwner = [sl \in Slots |-> {}],

    \* Per-writer retry counter for the active save / clear_shared slot attempt
    retryCount = [w \in Writers |-> 0],

    \* Shared budget of writer/clearer operations across all Writers
    opBudget = MaxOps,

    \* Latest accepted snapshot from try_load_snapshot (writers) and
    \* read_for_lookup (readers), validated by NoTornRead / BoundNoneNeverHits
    lastSnap = [w \in Writers |-> NoSnap],
    lastLookup = [r \in Readers |-> NoSnap];

define
    TypeOK ==
        /\ seq \in [Slots -> Nat]
        /\ board \in [Slots -> Boards]
        /\ meta \in [Slots -> MetaType]
        /\ \A sl \in Slots :
            \A s \in DOMAIN published[sl] :
                s \in Nat /\ s % 2 = 0 /\ published[sl][s] \in Boards \X MetaType
        /\ writeOwner \in [Slots -> SUBSET Writers]
        /\ retryCount \in [Writers -> 0..(MaxRetries + 1)]
        /\ opBudget \in 0..MaxOps
        /\ lastSnap \in [Writers -> SnapType]
        /\ lastLookup \in [Readers -> SnapType]

    \* At most one writer/clearer holds write access to any slot, and active
    \* write ownership corresponds strictly to an odd sequence number.
    SingleWriterPerSlot ==
        \A sl \in Slots :
            /\ Cardinality(writeOwner[sl]) <= 1
            /\ (writeOwner[sl] # {} <=> seq[sl] % 2 = 1)

    \* Every accepted lookup hit and every accepted save snapshot equals the
    \* exact coherent (board, meta) pair published at that even sequence number.
    NoTornRead ==
        /\ \A r \in Readers :
            lastLookup[r].valid =>
                /\ lastLookup[r].seq \in DOMAIN published[lastLookup[r].slot]
                /\ published[lastLookup[r].slot][lastLookup[r].seq] =
                       <<lastLookup[r].board, lastLookup[r].meta>>
        /\ \A w \in Writers :
            lastSnap[w].valid =>
                /\ lastSnap[w].seq \in DOMAIN published[lastSnap[w].slot]
                /\ published[lastSnap[w].slot][lastSnap[w].seq] =
                       <<lastSnap[w].board, lastSnap[w].meta>>

    \* lookup never returns a Bound::None (empty or cleared) slot, even when
    \* querying board = 0.
    BoundNoneNeverHits ==
        \A r \in Readers :
            lastLookup[r].valid => lastLookup[r].meta.bound # "None"

    \* save and clear_shared never exceed MaxRetries (SAVE_RETRIES = 4) per slot.
    BoundedRetries ==
        \A w \in Writers : retryCount[w] <= MaxRetries

    \* Liveness: every thread finishes all its bounded operations without
    \* deadlock or livelock.
    AllOpsFinished == \A p \in (Writers \cup Readers) : pc[p] = "Done"
    EventuallyDone == <>AllOpsFinished
end define;

macro ResetWriterLocals() begin
    req := NoReq;
    targetSlot := 0;
    s1 := 0;
    sb := 0;
    sm := EmptyMeta;
    retryCount[self] := 0;
end macro;

fair process Writer \in Writers
variables
    req = NoReq,
    targetSlot = 0,
    s1 = 0,
    sb = 0,
    sm = EmptyMeta;
begin
W_Loop:
    while opBudget > 0 do
        with op \in OpsFor(self) do
            req := op;
            opBudget := opBudget - 1;
            retryCount[self] := 0;
            lastSnap[self] := NoSnap;
            if op.kind = "Clear" then
                targetSlot := 0;
                goto C_Seq1;
            else
                targetSlot := PickProbeSlot(op.board, op.meta.gen, seq, board, meta);
                if op.meta.bound = "Exact" then
                    goto S_ExactSeq;
                else
                    goto S_SnapSeq1;
                end if;
            end if;
        end with;

    \* ---- save_exact: reads seq only before claiming the slot ----
S_ExactSeq:
        if (IF Mutant = "exceed_retry_budget"
            THEN retryCount[self] > MaxRetries
            ELSE retryCount[self] >= MaxRetries)
        then
            ResetWriterLocals();
            goto W_Loop;
        else
            retryCount[self] := retryCount[self] + 1;
            s1 := seq[targetSlot];
            if seq[targetSlot] % 2 = 1 then
                goto S_ExactSeq;
            else
                goto S_Claim;
            end if;
        end if;

    \* ---- non-Exact save: try_load_snapshot (seq1 -> meta -> board -> seq2) ----
S_SnapSeq1:
        if (IF Mutant = "exceed_retry_budget"
            THEN retryCount[self] > MaxRetries
            ELSE retryCount[self] >= MaxRetries)
        then
            ResetWriterLocals();
            goto W_Loop;
        else
            retryCount[self] := retryCount[self] + 1;
            s1 := seq[targetSlot];
            if seq[targetSlot] % 2 = 1 then
                goto S_SnapSeq1;
            else
                goto S_SnapMeta;
            end if;
        end if;

S_SnapMeta:
        sm := meta[targetSlot];
        if meta[targetSlot].bound = "None" then
            sb := 0;
            goto S_SnapSeq2;
        else
            goto S_SnapBoard;
        end if;

S_SnapBoard:
        sb := board[targetSlot];

S_SnapSeq2:
        if s1 # seq[targetSlot] then
            sb := 0;
            sm := EmptyMeta;
            goto S_SnapSeq1;
        else
            lastSnap[self] := [valid |-> TRUE, slot |-> targetSlot,
                               seq |-> s1, board |-> sb, meta |-> sm];
            if ~ShouldReplace(req.board, req.meta, sb, sm) then
                if Mutant = "livelock_reject_loop" then
                    \* Mutant: retry indefinitely instead of returning on reject
                    retryCount[self] := retryCount[self] - 1;
                    sb := 0;
                    sm := EmptyMeta;
                    goto S_SnapSeq1;
                else
                    ResetWriterLocals();
                    goto W_Loop;
                end if;
            else
                goto S_Claim;
            end if;
        end if;

    \* ---- seqlock_write: CAS even->odd, write board & meta, publish even ----
S_Claim:
        if seq[targetSlot] = s1 then
            seq[targetSlot] := s1 + 1;
            writeOwner[targetSlot] := writeOwner[targetSlot] \cup {self};
            sb := 0;
            sm := EmptyMeta;
            lastSnap[self] := NoSnap;
            goto S_WriteBoard;
        else
            sb := 0;
            sm := EmptyMeta;
            lastSnap[self] := NoSnap;
            if req.meta.bound = "Exact" then
                goto S_ExactSeq;
            else
                goto S_SnapSeq1;
            end if;
        end if;

S_WriteBoard:
        board[targetSlot] := req.board;
        if Mutant = "publish_before_payload" then
            \* Mutant: publish even seq before writing meta payload
            seq[targetSlot] := s1 + 2;
            published[targetSlot] :=
                published[targetSlot] @@ ((s1 + 2) :> <<req.board, req.meta>>);
            writeOwner[targetSlot] := writeOwner[targetSlot] \ {self};
        end if;

S_WriteMeta:
        meta[targetSlot] := req.meta;

S_Publish:
        if Mutant # "publish_before_payload" then
            seq[targetSlot] := s1 + 2;
            published[targetSlot] :=
                published[targetSlot] @@ ((s1 + 2) :> <<req.board, req.meta>>);
            writeOwner[targetSlot] := writeOwner[targetSlot] \ {self};
        end if;
        ResetWriterLocals();
        goto W_Loop;

    \* ---- TranspositionTable::clear_shared over slots 0 and 1 ----
C_Seq1:
        if (IF Mutant = "exceed_retry_budget"
            THEN retryCount[self] > MaxRetries
            ELSE retryCount[self] >= MaxRetries)
        then
            if targetSlot = 0 then
                targetSlot := 1;
                retryCount[self] := 0;
                s1 := 0;
                goto C_Seq1;
            else
                ResetWriterLocals();
                goto W_Loop;
            end if;
        else
            retryCount[self] := retryCount[self] + 1;
            if Mutant = "clear_without_cas" then
                \* Mutant: overwrite slot without odd-seq CAS claim
                writeOwner[targetSlot] := writeOwner[targetSlot] \cup {self};
                goto C_WriteBoard;
            else
                s1 := seq[targetSlot];
                if seq[targetSlot] % 2 = 1 then
                    goto C_Seq1;
                else
                    goto C_CheckEmpty;
                end if;
            end if;
        end if;

C_CheckEmpty:
        if meta[targetSlot] = EmptyMeta then
            goto C_VerifyEmpty;
        else
            goto C_Claim;
        end if;

C_VerifyEmpty:
        if seq[targetSlot] = s1 then
            \* Fast-skip already empty slot without dirtying it
            if targetSlot = 0 then
                targetSlot := 1;
                retryCount[self] := 0;
                s1 := 0;
                goto C_Seq1;
            else
                ResetWriterLocals();
                goto W_Loop;
            end if;
        else
            goto C_Seq1;
        end if;

C_Claim:
        if seq[targetSlot] = s1 then
            seq[targetSlot] := s1 + 1;
            writeOwner[targetSlot] := writeOwner[targetSlot] \cup {self};
            goto C_WriteBoard;
        else
            goto C_Seq1;
        end if;

C_WriteBoard:
        board[targetSlot] := 0;

C_WriteMeta:
        meta[targetSlot] := EmptyMeta;

C_Publish:
        if Mutant # "clear_without_cas" then
            seq[targetSlot] := s1 + 2;
            published[targetSlot] :=
                published[targetSlot] @@ ((s1 + 2) :> <<0, EmptyMeta>>);
        end if;
        writeOwner[targetSlot] := writeOwner[targetSlot] \ {self};
        if targetSlot = 0 then
            targetSlot := 1;
            retryCount[self] := 0;
            s1 := 0;
            goto C_Seq1;
        else
            ResetWriterLocals();
            goto W_Loop;
        end if;
    end while;
end process;

fair process Reader \in Readers
variables
    rOp = 0,
    targetBoard = 0,
    rSlot = 0,
    rSeq1 = 0,
    rBoard = 0,
    rMeta = EmptyMeta;
begin
R_Loop:
    while rOp < MaxReaderOps do
        with b \in Boards do
            targetBoard := b;
            rOp := rOp + 1;
            rSlot := 0;
            rSeq1 := 0;
            rBoard := 0;
            rMeta := EmptyMeta;
            lastLookup[self] := NoSnap;
            goto R_Seq1;
        end with;

R_Seq1:
        if seq[rSlot] % 2 = 1 /\ Mutant # "accept_odd_seq" then
            if rSlot = 0 then
                rSlot := 1;
                goto R_Seq1;
            else
                targetBoard := 0;
                rSlot := 0;
                goto R_Loop;
            end if;
        else
            rSeq1 := seq[rSlot];
            goto R_Board;
        end if;

R_Board:
        if board[rSlot] # targetBoard then
            \* Fast mismatch exit without seq2 validation (read_for_lookup)
            if rSlot = 0 then
                rSlot := 1;
                rSeq1 := 0;
                goto R_Seq1;
            else
                targetBoard := 0;
                rSlot := 0;
                rSeq1 := 0;
                goto R_Loop;
            end if;
        else
            rBoard := board[rSlot];
            goto R_Meta;
        end if;

R_Meta:
        rMeta := meta[rSlot];

R_Seq2:
        if (rSeq1 = seq[rSlot] \/ Mutant = "skip_seq2_check")
           /\ (rMeta.bound # "None" \/ Mutant = "ignore_bound_none")
        then
            lastLookup[self] := [valid |-> TRUE, slot |-> rSlot,
                                 seq |-> rSeq1, board |-> rBoard,
                                 meta |-> rMeta];
            targetBoard := 0;
            rSlot := 0;
            rSeq1 := 0;
            rBoard := 0;
            rMeta := EmptyMeta;
            goto R_Loop;
        else
            if rSlot = 0 then
                rSlot := 1;
                rSeq1 := 0;
                rBoard := 0;
                rMeta := EmptyMeta;
                goto R_Seq1;
            else
                targetBoard := 0;
                rSlot := 0;
                rSeq1 := 0;
                rBoard := 0;
                rMeta := EmptyMeta;
                goto R_Loop;
            end if;
        end if;
    end while;
end process;

end algorithm; *)
\* BEGIN TRANSLATION
\* END TRANSLATION
=============================================================================
