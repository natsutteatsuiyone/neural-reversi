---------------------------- MODULE ThreadPool ----------------------------
EXTENDS Integers, FiniteSets, Sequences, TLC

CONSTANTS
    N,
    MaxSearches,
    Mutant

ASSUME N >= 2 /\ MaxSearches >= 1

MainId == 0
Workers == 1..(N - 1)
Threads == 0..(N - 1)
CallerId == N
AllActors == Threads \cup {CallerId}
NONE == -1
SearchKinds == {"Midgame", "Endgame", "MidToEnd"}
Messages == SearchKinds \cup {"Exit"}
WorkModes == {"idle", "helper", "split"}

(* --algorithm ThreadPool

variables
    ready = [t \in Threads |-> (t = MainId)],
    poolReady = FALSE,
    thinking = FALSE,
    msgQueue = <<>>,
    resultReady = FALSE,
    searchesSent = 0,
    searching = [t \in Threads |-> FALSE],
    helperJob = [w \in Workers |-> FALSE],
    helperJobAborted = [w \in Workers |-> FALSE],
    workMode = [w \in Workers |-> "idle"],
    spActive = FALSE,
    helpersMask = {},
    activeSplit = [w \in Workers |-> FALSE],
    sleeping = [w \in Workers |-> FALSE],
    exitFlag = [t \in Threads |-> FALSE],
    exited = [t \in Threads |-> FALSE],
    exiting = FALSE,
    exitOwner = NONE,
    joining = [a \in AllActors |-> NONE],
    holdsRef = [a \in AllActors |-> (a = CallerId)];

define
    RefCount == Cardinality({a \in AllActors : holdsRef[a]})
    OtherRefs(a) == {x \in AllActors : x # a /\ holdsRef[x]}

    TypeOK ==
        /\ ready \in [Threads -> BOOLEAN]
        /\ poolReady \in BOOLEAN
        /\ thinking \in BOOLEAN
        /\ msgQueue \in Seq(Messages)
        /\ Len(msgQueue) <= 2
        /\ resultReady \in BOOLEAN
        /\ searchesSent \in 0..MaxSearches
        /\ searching \in [Threads -> BOOLEAN]
        /\ helperJob \in [Workers -> BOOLEAN]
        /\ helperJobAborted \in [Workers -> BOOLEAN]
        /\ workMode \in [Workers -> WorkModes]
        /\ spActive \in BOOLEAN
        /\ helpersMask \subseteq Threads
        /\ activeSplit \in [Workers -> BOOLEAN]
        /\ sleeping \in [Workers -> BOOLEAN]
        /\ exitFlag \in [Threads -> BOOLEAN]
        /\ exited \in [Threads -> BOOLEAN]
        /\ exiting \in BOOLEAN
        /\ exitOwner \in AllActors \cup {NONE}
        /\ joining \in [AllActors -> Threads \cup {NONE}]
        /\ holdsRef \in [AllActors -> BOOLEAN]

    \* Pool startup barrier: no search starts before all threads signal ready.
    ReadyBeforeSearch ==
        (poolReady \/ searchesSent > 0 \/ thinking) => \A t \in Threads : ready[t]

    \* No worker simultaneously holds or executes Lazy SMP helper work and YBWC
    \* split-point work, and no YBWC split point runs while Lazy SMP helpers are
    \* still booked or executing.
    NoHelperSplitOverlap ==
        /\ \A w \in Workers : ~(helperJob[w] /\ activeSplit[w])
        /\ \A w \in Workers : workMode[w] = "helper" => ~spActive /\ ~activeSplit[w]
        /\ \A w \in Workers : workMode[w] = "split" => ~helperJob[w]
        /\ spActive => \A w \in Workers : workMode[w] # "helper" /\ ~helperJob[w]

    \* ThreadPool::exit() never begins while any Arc<ThreadPool> reference is held.
    NoExitWhileRefHeld ==
        exiting => RefCount = 0

    \* No search, helper job, or split point is active or starts once exit begins.
    NoUseAfterExit ==
        exiting =>
            /\ ~thinking
            /\ ~spActive
            /\ \A w \in Workers : workMode[w] = "idle" /\ ~helperJob[w] /\ ~activeSplit[w]

    \* A thread never attempts to join() itself during shutdown (pool.rs:174-179).
    NoSelfJoin ==
        \A t \in Threads : joining[t] # t

    \* Liveness: once all searches finish and the external ref is dropped, every
    \* thread in the pool reaches the exited state.
    AllExited == \A t \in Threads : exited[t]
    EventuallyExits == <>AllExited
end define;

procedure RunExit()
variables j = 0;
begin
EX_Enter:
    exiting := TRUE;
    exitOwner := self;
    exitFlag := [t \in Threads |-> TRUE];
EX_Notify:
    if Mutant # "exit_skips_notify" then
        sleeping := [w \in Workers |-> FALSE];
    end if;
    if Mutant # "exit_skips_main_msg" then
        msgQueue := Append(msgQueue, "Exit");
    end if;
EX_JoinLoop:
    while j < N do
        if Mutant = "drop_last_ref_self_join" \/ j # self then
            joining[self] := j;
EX_JoinWait:
            await exited[j];
            joining[self] := NONE;
        end if;
EX_Next:
        j := j + 1;
    end while;
EX_Done:
    return;
end procedure;

fair process Caller = CallerId
begin
C_WaitReady:
    if Mutant # "skip_wait_ready" then
        await \A t \in Threads : ready[t];
    end if;
    poolReady := TRUE;
C_Loop:
    if searchesSent < MaxSearches then
        either
            with k \in SearchKinds do
                thinking := TRUE;
                holdsRef[MainId] := TRUE;
                msgQueue := Append(msgQueue, k);
                searchesSent := searchesSent + 1;
            end with;
            goto C_AfterSend;
        or
            goto C_DropRef;
        end either;
    else
        goto C_DropRef;
    end if;
C_AfterSend:
    either
        await resultReady;
        resultReady := FALSE;
        goto C_Loop;
    or
        goto C_DropRef;
    end either;
C_DropRef:
    holdsRef[CallerId] := FALSE;
    if Mutant = "exit_before_zero_refs" \/ OtherRefs(CallerId) = {} then
        call RunExit();
    end if;
C_Done:
    skip;
end process;

fair process Main = MainId
variables curMsg = "Exit";
begin
M_Loop:
    if exitFlag[MainId] then
        goto M_Exit;
    end if;
M_Recv:
    await Len(msgQueue) > 0;
    curMsg := Head(msgQueue);
    msgQueue := Tail(msgQueue);
    if curMsg = "Exit" then
        exitFlag[MainId] := TRUE;
        goto M_Exit;
    else
        searching[MainId] := TRUE;
        \* pool.notify_all() on StartThinking (thread.rs:654)
        sleeping := [w \in Workers |-> FALSE];
    end if;
M_Dispatch:
    if curMsg \in {"Midgame", "MidToEnd"} then
        assert \A w \in Workers : ~searching[w];
        helperJob := [w \in Workers |-> TRUE];
        searching := [t \in Threads |-> TRUE];
        goto M_StopHelpers;
    else
        goto M_StartSplit;
    end if;
M_StopHelpers:
    if Mutant = "skip_stop_helpers_before_split" /\ curMsg = "MidToEnd" then
        goto M_StartSplit;
    else
        helperJobAborted := [w \in Workers |-> TRUE];
    end if;
M_WaitHelpers:
    await \A w \in Workers : ~searching[w];
    helperJobAborted := [w \in Workers |-> FALSE];
    if curMsg = "MidToEnd" then
        goto M_StartSplit;
    else
        goto M_FinishSearch;
    end if;
M_StartSplit:
    spActive := TRUE;
    with idle = {w \in Workers : ~searching[w]} do
        helpersMask := {MainId} \cup idle;
        activeSplit := [w \in Workers |-> IF w \in idle THEN TRUE ELSE activeSplit[w]];
        searching := [t \in Threads |-> IF t \in idle THEN TRUE ELSE searching[t]];
    end with;
M_MainSplitDone:
    helpersMask := helpersMask \ {MainId};
M_WaitSplit:
    await helpersMask = {};
    spActive := FALSE;
M_FinishSearch:
    searching[MainId] := FALSE;
    if Mutant # "drop_pool_before_clear_thinking" then
        thinking := FALSE;
    end if;
    holdsRef[MainId] := FALSE;
    if OtherRefs(MainId) = {} then
        call RunExit();
    end if;
M_SendResult:
    if Mutant = "drop_pool_before_clear_thinking" then
        thinking := FALSE;
    end if;
    resultReady := TRUE;
    goto M_Loop;
M_Exit:
    exited[MainId] := TRUE;
end process;

fair process Worker \in Workers
begin
W_Loop:
    if Mutant # "worker_ignores_exit_flag" /\ exitFlag[self] then
        goto W_Exit;
    elsif searching[self] then
        if helperJob[self] then
            helperJob[self] := FALSE;
            workMode[self] := "helper";
            goto W_RunHelper;
        else
            assert activeSplit[self];
            workMode[self] := "split";
            goto W_RunSplit;
        end if;
    elsif ~thinking then
        goto W_SleepCheck;
    else
        await searching[self] \/ ~thinking \/ exitFlag[self];
        goto W_Loop;
    end if;
W_RunHelper:
    workMode[self] := "idle";
    searching[self] := FALSE;
    goto W_Loop;
W_RunSplit:
    activeSplit[self] := FALSE;
    workMode[self] := "idle";
    searching[self] := FALSE;
    helpersMask := helpersMask \ {self};
W_TryLateJoin:
    \* Models self.pool.upgrade() in try_late_join() (thread.rs:715):
    \* retains an Arc<ThreadPool> after clearing helpers_mask, so this worker
    \* may drop the last pool reference after root search returns.
    if RefCount > 0 then
        holdsRef[self] := TRUE;
    else
        goto W_Loop;
    end if;
W_DropLateJoin:
    holdsRef[self] := FALSE;
    if OtherRefs(self) = {} then
        call RunExit();
    end if;
W_AfterDrop:
    goto W_Loop;
W_SleepCheck:
    \* Under mutex_for_sleep_condition (thread.rs:521-525):
    \* publish ready and enter sleep_condition.wait iff !exit && !thinking.
    ready[self] := TRUE;
    if ~exitFlag[self] /\ ~thinking then
        sleeping[self] := TRUE;
        goto W_Wait;
    else
        goto W_Loop;
    end if;
W_Wait:
    await ~sleeping[self];
    goto W_Loop;
W_Exit:
    exited[self] := TRUE;
end process;

end algorithm; *)
\* BEGIN TRANSLATION
\* END TRANSLATION
=============================================================================
