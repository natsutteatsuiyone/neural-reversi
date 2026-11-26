---------------------------- MODULE SplitPoint ----------------------------
EXTENDS Integers, FiniteSets, Sequences, TLC

CONSTANTS
    N,
    MaxSP,
    Moves,
    Budget,
    MaxThr,
    MaxUnwinds,
    Mutant

Threads == 0..(N - 1)
Workers == 1..(N - 1)
SPs == Threads \X (0..(MaxSP - 1))
NONE == <<-1, -1>>
FREE == -1
TimerId == N
Owner(sp) == sp[1]

RECURSIVE ChainOf(_, _, _)
ChainOf(sp, par, d) ==
    IF sp = NONE \/ d = 0 THEN {} ELSE {sp} \cup ChainOf(par[sp], par, d - 1)

(* --algorithm SplitPoint

variables
    searching = [t \in Threads |-> FALSE],
    splitPointsSize = [t \in Threads |-> 0],
    active = [t \in Threads |-> NONE],
    seenEpoch = [t \in Threads |-> 0],
    chainCut = [t \in Threads |-> FALSE],
    spLock = [s \in SPs |-> FREE],
    task = [s \in SPs |-> FALSE],
    parent = [s \in SPs |-> NONE],
    slotGeneration = [s \in SPs |-> 0],
    helpersMask = [s \in SPs |-> {}],
    allHelpersSearching = [s \in SPs |-> FALSE],
    cutoff = [s \in SPs |-> FALSE],
    level = [s \in SPs |-> 0],
    nextMove = [s \in SPs |-> 0],
    epoch = 0,
    thinking = TRUE,
    aborted = FALSE,
    budget = Budget,
    failed = [t \in Threads |-> FALSE],
    unwinding = [t \in Threads |-> FALSE],
    unwindsLeft = MaxUnwinds,
    observedReading = [t \in Threads |-> {}],
    cutoffBumpPending = [s \in SPs |-> FALSE];

define
    Chain(sp) == ChainOf(sp, parent, N * MaxSP)
    Deepest(t) == <<t, splitPointsSize[t] - 1>>

    CanJoin(t, sp) ==
        /\ Mutant = "can_join_ignores_searching" \/ ~searching[t]
        /\ Mutant = "unwind_can_join_ignores_failed" \/ ~failed[t]
        /\ IF splitPointsSize[t] = 0 THEN TRUE ELSE Owner(sp) \in helpersMask[Deepest(t)]

    CanSplit(t) ==
        /\ N > 1
        /\ splitPointsSize[t] < MaxSP
        /\ IF active[t] = NONE THEN TRUE
           ELSE \/ ~allHelpersSearching[active[t]]
                \/ N > MaxThr /\ Cardinality(helpersMask[active[t]]) = MaxThr

    LateJoinCandidate(t, sp) ==
        /\ ~cutoff[sp]
        /\ Cardinality(helpersMask[sp]) < MaxThr
        /\ allHelpersSearching[sp]
        /\ CanJoin(t, sp)

    AcceptsHelpers(sp) ==
        IF Mutant = "late_join_no_recheck" THEN TRUE
        ELSE ~cutoff[sp] /\ allHelpersSearching[sp] /\ Cardinality(helpersMask[sp]) < MaxThr

    ChainCutNow(t) ==
        IF Mutant = "cutoff_check_only_own_split_point"
        THEN active[t] # NONE /\ cutoff[active[t]]
        ELSE \E sp \in Chain(active[t]) : cutoff[sp]

    ChainIsCut(t) == \E sp \in Chain(active[t]) : cutoff[sp]

    UsedBy(t) == {u[1] : u \in observedReading[t]}

    TaskAliveWhileUsed ==
        \A t \in Threads : \A u \in observedReading[t] : task[u[1]] /\ slotGeneration[u[1]] = u[2]

    BookedOnlyIntoLiveSplitPoints ==
        \A sp \in SPs : helpersMask[sp] # {} => task[sp]

    BookingMatchesThread ==
        \A t \in Threads, sp \in SPs :
            (t \in helpersMask[sp] /\ sp \notin UsedBy(t) /\ ~failed[t])
                => active[t] = sp /\ searching[t]

    ActiveChainAlive ==
        \A t \in Threads : \A sp \in Chain(active[t]) : task[sp]

    CutoffCacheSound ==
        \A t \in Threads :
            (active[t] # NONE /\ seenEpoch[t] = epoch /\ ~chainCut[t])
                => \A sp \in Chain(active[t]) : cutoff[sp] => cutoffBumpPending[sp]

    CutoffCacheNoFalseStop ==
        \A t \in Threads : (active[t] # NONE /\ chainCut[t]) => ChainIsCut(t)

    CleanTeardown ==
        ~thinking =>
            /\ \A sp \in SPs :
                ~task[sp] /\ parent[sp] = NONE /\ helpersMask[sp] = {}
                /\ (spLock[sp] = FREE \/ pc[spLock[sp]] = "SA_Unlock")
                /\ ~cutoffBumpPending[sp]
            /\ \A t \in Threads :
                ~searching[t] /\ splitPointsSize[t] = 0 /\ active[t] = NONE /\ observedReading[t] = {}

    ParentChainAcyclic ==
        \A sp \in SPs : sp \notin Chain(parent[sp])

    SplitBudgetRespected == budget >= 0

    UnwindBudgetRespected == unwindsLeft >= 0
end define;

macro ResetCutoffCache(t) begin
    seenEpoch[t] := epoch - 1;
    chainCut[t] := FALSE;
end macro;

macro BookInto(t, sp) begin
    helpersMask[sp] := helpersMask[sp] \cup {t};
    active[t] := sp;
    searching[t] := TRUE;
end macro;

macro FinalizeSplit(sp) begin
    assert ~searching[self];
    searching[self] := TRUE;
    splitPointsSize[self] := splitPointsSize[self] - 1;
    active[self] := parent[sp];
end macro;

procedure AbandonActiveSplit()
variables usp = NONE;
begin
UA_Detach:
    failed[self] := TRUE;
    usp := active[self];
    if Mutant # "unwind_keeps_active" then
        active[self] := NONE;
    end if;
    ResetCutoffCache(self);
    searching[self] := FALSE;
UA_Release:
    if usp # NONE then
        await spLock[usp] = FREE;
        if Owner(usp) # self then
            aborted := TRUE;
        end if;
        allHelpersSearching[usp] := FALSE;
        if Mutant # "unwind_skips_helper_bit_reset" then
            helpersMask[usp] := helpersMask[usp] \ {self};
        end if;
    end if;
    return;
end procedure;

procedure Split()
variables nsp = NONE, i = 0;
begin
SP_Init:
    with x = <<self, splitPointsSize[self]>> do
        await spLock[x] = FREE;
        assert searching[self];
        nsp := x;
        spLock[x] := self;
        task[x] := TRUE;
        parent[x] := active[self];
        slotGeneration[x] := slotGeneration[x] + 1;
        level[x] := IF active[self] = NONE THEN 0 ELSE level[active[self]] + 1;
        helpersMask[x] := {self};
        allHelpersSearching[x] := TRUE;
        cutoff[x] := FALSE;
        cutoffBumpPending[x] := FALSE;
        nextMove[x] := 0;
        splitPointsSize[self] := splitPointsSize[self] + 1;
        active[self] := x;
        ResetCutoffCache(self);
    end with;
SP_Assign:
    while i < N do
        if Cardinality(helpersMask[nsp]) >= MaxThr then
            goto SP_Unlock;
        end if;
SP_Book:
        if CanJoin(i, nsp) then
            BookInto(i, nsp);
        end if;
        i := i + 1;
    end while;
SP_Unlock:
    if Mutant # "hold_lock_across_idle_loop" then
        spLock[nsp] := FREE;
    end if;
SP_Wait:
    either
        call IdleLoop(nsp);
    or
        await unwindsLeft > 0;
        unwindsLeft := unwindsLeft - 1;
        unwinding[self] := TRUE;
    end either;
SP_Finalize:
    if unwinding[self] then
        call AbandonActiveSplit();
SP_UnwindDrain:
        if Mutant # "unwind_clears_slot_before_helpers_leave" then
            await helpersMask[nsp] = {};
        end if;
        FinalizeSplit(nsp);
    else
        FinalizeSplit(nsp);
    end if;
SP_Clear:
    if Mutant # "finalize_skips_clear" then
        task[nsp] := FALSE;
        parent[nsp] := NONE;
    end if;
    return;
end procedure;

procedure IdleLoop(thisSp)
begin
IL_Top:
    await \/ thisSp # NONE /\ helpersMask[thisSp] = {}
          \/ searching[self]
          \/ thisSp = NONE /\ ~thinking;
    if thisSp # NONE /\ helpersMask[thisSp] = {} then
        return;
    end if;
IL_Inner:
    if unwinding[self] then
        return;
    elsif searching[self] then
        call SearchActive();
        goto IL_Inner;
    end if;
IL_Sleep:
    if thisSp = NONE /\ ~thinking then
        return;
    else
        goto IL_Top;
    end if;
end procedure;

procedure SearchActive()
variables sa = NONE;
begin
SA_ReadActive:
    assert active[self] # NONE;
    sa := active[self];
SA_ReadTask:
    assert task[sa];
    observedReading[self] := observedReading[self] \cup {<<sa, slotGeneration[sa]>>};
SA_Search:
    call SearchSP(sa);
SA_Lock:
    if unwinding[self] then
        observedReading[self] := {u \in observedReading[self] : u[1] # sa};
        call AbandonActiveSplit();
SA_UnwindReturn:
        return;
    else
        await spLock[sa] = FREE;
        spLock[sa] := self;
    end if;
SA_Detach:
    if Mutant # "leave_keeps_active" then
        active[self] := NONE;
    end if;
    if Mutant # "leave_no_cache_reset" then
        ResetCutoffCache(self);
    end if;
    searching[self] := FALSE;
    if Mutant # "leave_keeps_all_helpers_searching" then
        allHelpersSearching[sa] := FALSE;
    end if;
    helpersMask[sa] := helpersMask[sa] \ {self};
    if Mutant # "reset_bit_before_merge_counters" then
        observedReading[self] := {u \in observedReading[self] : u[1] # sa};
    end if;
SA_Unlock:
    if Mutant = "reset_bit_before_merge_counters" then
        observedReading[self] := {u \in observedReading[self] : u[1] # sa};
    end if;
    spLock[sa] := FREE;
SA_LateJoin:
    call TryLateJoin();
    return;
end procedure;

procedure SearchSP(ss)
variables obsEpoch = 0;
begin
SS_Next:
    if nextMove[ss] >= Moves then
        return;
    else
        nextMove[ss] := nextMove[ss] + 1;
    end if;
SS_Child:
    either
        skip;
    or
        await CanSplit(self) /\ budget > 0;
        budget := budget - 1;
        call Split();
    or
        await unwindsLeft > 0;
        unwindsLeft := unwindsLeft - 1;
        unwinding[self] := TRUE;
        return;
    end either;
SS_CheckEpoch:
    if unwinding[self] then
        return;
    elsif ~chainCut[self] /\ epoch # seenEpoch[self] then
        obsEpoch := epoch;
        goto SS_WalkChain;
    else
        goto SS_Publish;
    end if;
SS_WalkChain:
    seenEpoch[self] := obsEpoch;
    chainCut[self] := ChainCutNow(self);
    obsEpoch := 0;
SS_Publish:
    await spLock[ss] = FREE;
    with refresh = ~chainCut[self] /\ epoch # seenEpoch[self],
         stop = chainCut[self] \/ (refresh /\ ChainCutNow(self)) \/ aborted do
        if refresh then
            seenEpoch[self] := epoch;
            chainCut[self] := ChainCutNow(self);
        end if;
        if stop then
            allHelpersSearching[ss] := FALSE;
            return;
        else
            either
                goto SS_Next;
            or
                if cutoff[ss] then
                    return;
                else
                    cutoff[ss] := TRUE;
                    cutoffBumpPending[ss] := TRUE;
                    spLock[ss] := self;
                    goto SS_Bump;
                end if;
            end either;
        end if;
    end with;
SS_Bump:
    epoch := epoch + 1;
    cutoffBumpPending[ss] := FALSE;
    spLock[ss] := FREE;
    return;
end procedure;

procedure TryLateJoin()
variables cand = NONE;
begin
TJ_Scan:
    with ok = {Deepest(t) : t \in {u \in Threads : splitPointsSize[u] > 0}} do
        with c = {sp \in ok : LateJoinCandidate(self, sp)} do
            if c = {} then
                return;
            else
                with x \in {y \in c : \A z \in c : level[y] <= level[z]} do
                    cand := x;
                end with;
            end if;
        end with;
    end with;
TJ_Lock:
    await spLock[cand] = FREE;
    if ~AcceptsHelpers(cand) then
        return;
    else
        spLock[cand] := self;
    end if;
TJ_Book:
    if CanJoin(self, cand) then
        BookInto(self, cand);
    end if;
    spLock[cand] := FREE;
    return;
end procedure;

process Main = 0
begin
M_Start:
    searching[0] := TRUE;
M_Root:
    either
        await ~unwinding[0] /\ CanSplit(0) /\ budget > 0;
        budget := budget - 1;
        call Split();
        goto M_Root;
    or
        skip;
    end either;
M_End:
    searching[0] := FALSE;
    thinking := FALSE;
end process;

process Worker \in Workers
begin
W_Start:
    call IdleLoop(NONE);
end process;

process Timer = TimerId
begin
T_Fire:
    either
        aborted := TRUE;
    or
        skip;
    end either;
end process;

end algorithm; *)
\* BEGIN TRANSLATION
\* END TRANSLATION
=============================================================================
