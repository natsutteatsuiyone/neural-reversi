----------------------------- MODULE StopAbort -----------------------------
EXTENDS Integers, FiniteSets, TLC

CONSTANTS
    GenMod,
    MaxRuns,
    NumTokens,
    Mutant

ASSUME GenMod >= 2 /\ MaxRuns > GenMod /\ NumTokens >= 1

Gens == 0..(GenMod - 1)
NO_TOKEN == 0
Tokens == 1..NumTokens
RunnerId == 0
TimerId == NumTokens + 1
StaleCallerId == NumTokens + 2
TimerModes == {"quiet", "fire"}

(* --algorithm StopAbort

variables
    \* AbortState atomic word: (gen, abortGen) in Gens x Gens.
    \* Matches AbortState::new() = AtomicU64::new(ABORT_GEN_MASK):
    \* high 32 bits = 0, low 32 bits = u32::MAX (GenMod - 1).
    gen = 0,
    abortGen = GenMod - 1,

    \* StopToken mutex-protected state per token:
    \* stopped flag and set of attached <<generation, originRunId>> entries.
    tokenStopped = [t \in Tokens |-> FALSE],
    tokenRuns = [t \in Tokens |-> {}],

    \* ThreadPool timer coordination: timer_stop flag and live timer threads.
    \* Each live timer thread carries <<armedGen, originRunId, mode>>.
    timerStop = FALSE,
    activeTimers = {},

    \* Consecutive search run bookkeeping (ghost + runner state).
    runId = 0,
    runActive = FALSE,
    runAttached = FALSE,
    runTok = NO_TOKEN,
    runAbortAuthorized = FALSE,
    staleLeak = FALSE,
    allRunsDone = FALSE;

define
    IsAborted == gen = abortGen

    PrevGen == (gen + GenMod - 1) % GenMod
    PrevRun == runId - 1

    TypeOK ==
        /\ gen \in Gens
        /\ abortGen \in Gens
        /\ tokenStopped \in [Tokens -> BOOLEAN]
        /\ tokenRuns \in [Tokens -> SUBSET (Gens \X (1..MaxRuns))]
        /\ timerStop \in BOOLEAN
        /\ activeTimers \subseteq (Gens \X (1..MaxRuns) \X TimerModes)
        /\ runId \in 0..MaxRuns
        /\ runActive \in BOOLEAN
        /\ runAttached \in BOOLEAN
        /\ runTok \in Tokens \cup {NO_TOKEN}
        /\ (runAttached => runTok \in Tokens)
        /\ (~runAttached => runTok = NO_TOKEN)
        /\ runAbortAuthorized \in BOOLEAN
        /\ staleLeak \in BOOLEAN
        /\ allRunsDone \in BOOLEAN

    \* No StopToken::stop() entry from a prior run, no delayed timer from a
    \* prior run, and no stale request_abort_for(prev_gen) ever aborts a
    \* different/subsequent run.
    NoStaleAbortLeak == ~staleLeak

    \* Whenever no abort was requested for the active run (neither via its
    \* attached StopToken nor via its own timer), AbortState::is_aborted()
    \* must be FALSE — even across generation wraparound.
    FreshRunNotSpuriouslyAborted ==
        (runActive /\ ~runAbortAuthorized) => ~IsAborted

    \* Once StopToken::stop() has completed and an active run is attached to
    \* that token (whether stop() ran before or after attach()), the current
    \* run's generation is aborted in AbortState.
    StoppedTokenReachesAttached ==
        (runActive /\ runAttached /\ tokenStopped[runTok]) => IsAborted

    \* Liveness: every search run completes, detaches, and joins its timer.
    AllRunsDone == allRunsDone /\ activeTimers = {}
    EventuallyTerminates == <>AllRunsDone
end define;

macro RequestAbortFor(targetGen, originRun) begin
    if Mutant = "request_abort_for_ignores_gen" then
        \* Bug: ignores targetGen and aborts whatever generation is current.
        abortGen := gen;
        if runActive /\ originRun = runId then
            runAbortAuthorized := TRUE;
        elsif runActive /\ originRun # runId /\ ~runAbortAuthorized then
            staleLeak := TRUE;
        end if;
    elsif Mutant = "request_abort_for_clobbers_abort" then
        \* Bug: removes `(state >> 32) as u32 == generation` guard in abort.rs,
        \* overwriting abort_gen with targetGen unconditionally.
        abortGen := targetGen;
        if targetGen = gen then
            if runActive /\ originRun = runId then
                runAbortAuthorized := TRUE;
            elsif runActive /\ originRun # runId /\ ~runAbortAuthorized then
                staleLeak := TRUE;
            end if;
        end if;
    else
        if gen = targetGen then
            abortGen := targetGen;
            if runActive /\ originRun = runId then
                runAbortAuthorized := TRUE;
            elsif runActive /\ originRun # runId /\ ~runAbortAuthorized then
                staleLeak := TRUE;
            end if;
        end if;
    end if;
end macro;

macro RequestAbort(originRun) begin
    abortGen := gen;
    if runActive /\ originRun = runId then
        runAbortAuthorized := TRUE;
    elsif runActive /\ originRun # runId /\ ~runAbortAuthorized then
        staleLeak := TRUE;
    end if;
end macro;

\* Consecutive search driver modeling Search::execute_search,
\* ThreadPool::start_thinking / start_timer / stop_timer, and
\* Thread::main_thread_loop (including StopAttachment::drop).
fair process Runner = RunnerId
begin
R_Begin:
    while runId < MaxRuns do
        \* AbortState::begin_search()
        with oldGen = gen, newGen = (gen + 1) % GenMod do
            gen := newGen;
            if Mutant # "wrap_keeps_old_abort_marker" then
                abortGen := oldGen;
            end if;
            runId := runId + 1;
            runActive := TRUE;
            runAttached := FALSE;
            runTok := NO_TOKEN;
            runAbortAuthorized := FALSE;
        end with;
R_Attach:
        \* Optional StopToken::attach(&abort_state, generation)
        with tok \in Tokens \cup {NO_TOKEN} do
            if tok # NO_TOKEN then
                runTok := tok;
                runAttached := TRUE;
                tokenRuns[tok] := tokenRuns[tok] \cup {<<gen, runId>>};
                if Mutant # "attach_skips_pre_stopped_check" /\ tokenStopped[tok] then
                    RequestAbortFor(gen, runId);
                end if;
            end if;
        end with;
R_StartTimer:
        \* Optional ThreadPool::start_timer(time_manager, generation)
        with mode \in {"none", "quiet", "fire"} do
            if mode # "none" then
                timerStop := FALSE;
                activeTimers := activeTimers \cup {<<gen, runId, mode>>};
            end if;
        end with;
R_Search:
        \* Main thread runs search_root and polls AbortState::is_aborted().
        skip;
R_Detach:
        \* Main thread drops StopAttachment before sending SearchResult.
        if runAttached then
            if Mutant # "detach_does_not_remove" then
                tokenRuns[runTok] := tokenRuns[runTok] \ {<<gen, runId>>};
            end if;
            runAttached := FALSE;
            runTok := NO_TOKEN;
        end if;
        runActive := FALSE;
        runAbortAuthorized := FALSE;
R_StopTimerFlag:
        \* Caller receives SearchResult and calls ThreadPool::stop_timer():
        \* step 1 stores timer_stop = true.
        if Mutant # "stop_timer_skips_flag" then
            timerStop := TRUE;
        end if;
R_StopTimerJoin:
        \* Step 2 joins timer_handle (blocks until the timer thread exits).
        if Mutant \notin {"skip_timer_join", "unscoped_timer_abort"} then
            await activeTimers = {};
        end if;
    end while;
R_Finish:
    allRunsDone := TRUE;
end process;

\* Timer thread(s) modeling ThreadPool::timer_loop.
fair process Timer = TimerId
begin
T_Step:
    while ~allRunsDone \/ activeTimers # {} do
        await (\E t \in activeTimers : t[3] = "fire" \/ timerStop)
              \/ (allRunsDone /\ activeTimers = {});
        if activeTimers # {} then
            with t \in {x \in activeTimers : x[3] = "fire" \/ timerStop} do
                if t[3] = "quiet" then
                    \* Deadline not reached; observed timer_stop = true and exit.
                    activeTimers := activeTimers \ {t};
                else
                    either
                        \* Deadline reached: call TimeManager::check_time_for(generation)
                        \* (or unscoped request_abort in unscoped_timer_abort mutant).
                        if Mutant = "unscoped_timer_abort" then
                            RequestAbort(t[2]);
                        else
                            RequestAbortFor(t[1], t[2]);
                        end if;
                        activeTimers := activeTimers \ {t};
                    or
                        \* Observed timer_stop = true before deadline expired.
                        await timerStop;
                        activeTimers := activeTimers \ {t};
                    end either;
                end if;
            end with;
        end if;
    end while;
end process;

\* External callers that may invoke StopToken::stop() at any point
\* (before attach, during a run, after detach, after wraparound, or never).
fair process StopCaller \in Tokens
begin
SC_Act:
    either
        tokenStopped[self] := TRUE;
        with S = tokenRuns[self] do
            if Mutant = "request_abort_for_ignores_gen" then
                if S # {} then
                    abortGen := gen;
                    if runActive /\ \E e \in S : e[2] = runId then
                        runAbortAuthorized := TRUE;
                    elsif runActive /\ ~runAbortAuthorized /\ \E e \in S : e[2] # runId then
                        staleLeak := TRUE;
                    end if;
                end if;
            elsif Mutant = "request_abort_for_clobbers_abort" then
                if S # {} then
                    with e \in S do
                        abortGen := e[1];
                        if e[1] = gen then
                            if runActive /\ e[2] = runId then
                                runAbortAuthorized := TRUE;
                            elsif runActive /\ ~runAbortAuthorized /\ e[2] # runId then
                                staleLeak := TRUE;
                            end if;
                        end if;
                    end with;
                end if;
            else
                if \E e \in S : e[1] = gen then
                    abortGen := gen;
                    if runActive /\ \E e \in S : e[1] = gen /\ e[2] = runId then
                        runAbortAuthorized := TRUE;
                    elsif runActive /\ ~runAbortAuthorized /\ \E e \in S : e[1] = gen /\ e[2] # runId then
                        staleLeak := TRUE;
                    end if;
                end if;
            end if;
        end with;
    or
        skip;
    end either;
end process;

\* Delayed generation-scoped abort call targeting the immediately preceding
\* search generation (e.g., an in-flight AbortState::request_abort_for(prev_gen)
\* racing with the next begin_search(), as contracted in abort.rs).
fair process StaleAbortCaller = StaleCallerId
begin
SA_Act:
    await (runId >= 2 /\ runActive) \/ allRunsDone;
    if runId >= 2 /\ runActive then
        either
            RequestAbortFor(PrevGen, PrevRun);
        or
            skip;
        end either;
    end if;
end process;

end algorithm; *)
\* BEGIN TRANSLATION
\* END TRANSLATION
=============================================================================
