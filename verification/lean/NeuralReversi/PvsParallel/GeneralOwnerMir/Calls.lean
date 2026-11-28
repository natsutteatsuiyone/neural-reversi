module
public import NeuralReversi.PvsParallel.GeneralOwnerMir

@[expose] public section

namespace NeuralReversi.PvsParallel.GeneralOwnerMir.Calls

open PvsPostpone (Queue step shouldPostpone)
open Threading.SplitWindow (State)

structure Trace (Move : Type) where
  calls : List (Move × Int)
  remaining : Queue Move
  outcome : Prefix.Serial.Outcome
  bestMove : Option Move

def Trace.receipt (t : Trace Move) : Receipt Move :=
  ⟨t.calls.map Prod.fst,t.remaining,t.outcome,t.bestMove⟩

/-- The physical iterator starts at idx. Once it passes fresh_end all remaining
    elements are previously postponed moves. -/
def slots (idx fresh : Nat) (pending : List Move) : Queue Move :=
  ⟨idx,pending.take (fresh-idx),pending.drop (fresh-idx)⟩

@[simp] private theorem pending_slots (idx fresh : Nat) (pending : List Move) :
    (slots idx fresh pending).pending = pending := List.take_append_drop _ _

private theorem step_rotate (idx fresh : Nat) (m : Move) (rest : List Move)
    (fits : fresh-idx ≤ (m::rest).length) (rotate : shouldPostpone busy idx fresh = true) :
    step busy (slots idx fresh (m::rest)) = some (none,slots idx (fresh-1) (rest++[m])) := by
  obtain ⟨⟨_,_⟩,later⟩ := (by simpa [shouldPostpone] using rotate :
    (busy = true ∧ idx ≠ 0) ∧ idx+1 < fresh)
  obtain ⟨k,h⟩ : ∃ k, fresh-idx = k+1 := ⟨fresh-idx-1,by omega⟩
  have newCut : fresh - 1 - idx = k := by omega
  have inside : k ≤ rest.length := by simp only [List.length_cons] at fits; omega
  have cursor : idx + (min k rest.length + 1) = fresh := by omega
  simp [step,slots,h,newCut,cursor,rotate,List.take_append_of_le_length inside,
    List.drop_append_of_le_length inside]

private theorem step_take (idx fresh : Nat) (m : Move) (rest : List Move)
    (fits : fresh-idx ≤ (m::rest).length) (take : shouldPostpone busy idx fresh = false) :
    step busy (slots idx fresh (m::rest)) = some (some m,slots (idx+1) fresh rest) := by
  cases h : fresh - idx with
  | zero =>
    have zero : fresh-(idx+1) = 0 := by omega
    simp [slots,step,h,zero]
  | succ k =>
    have newCut : fresh-(idx+1) = k := by omega
    have cursor : idx + (min k rest.length + 1) = fresh := by
      simp only [List.length_cons] at fits
      omega
    simp [step,slots,h,newCut,cursor,take]

/-- Each processed move contributes its actual child word, even when the return
    depends on the iteration. Postponing rotates the unprocessed suffix in place. -/
def walk (schedule : Schedule Move) (source : Nat → Move → Int) (depth : Nat) (cut : Bool)
    (i idx fresh : Nat) (pending : List Move) (s : State) (previous : Option Move) : Trace Move :=
  let q := slots idx fresh pending
  match pending with
  | [] => ⟨[],q,.returned s.best,previous⟩
  | m::rest =>
    if canSplit idx (idx+(m::rest).length) depth cut (schedule.available i q) then
      ⟨[],q,.handoff s,previous⟩
    else if rotate : shouldPostpone (schedule.busy i q) idx fresh then
      walk schedule source depth cut (i+1) idx (fresh-1) (rest++[m]) s previous
    else
      let word := source i m
      let update := after s.alpha s.best (-word)
      let chosen := if update.2.1 then some m else previous
      if update.2.2 then ⟨[(m,word)],slots (idx+1) fresh rest,.returned (-word),chosen⟩
      else
        let tail := walk schedule source depth cut (i+1) (idx+1) fresh rest ⟨s.alpha,update.1⟩ chosen
        { tail with calls := (m,word)::tail.calls }
termination_by pending.length + (fresh-idx)
decreasing_by
  · simp only [shouldPostpone,Bool.and_eq_true,decide_eq_true_eq] at rotate
    simp only [List.length_append,List.length_cons,List.length_nil]
    omega
  · simp only [List.length_cons]; omega

/-- Execute one MIR header using the full move array and its original fresh_end.
    The returned index controls both the next array read and the split suffix. -/
theorem array_header (schedule : Schedule Move) (source : Nat → Move → Int) (depth : Nat) (cut : Bool)
    (i idx fresh : Nat) (moves : List Move) (s : State) (previous : Option Move) :
    let q := slots idx fresh (moves.drop idx)
    let word := (moves[idx]?.map (source i)).getD 0
    walk schedule source depth cut i idx fresh (moves.drop idx) s previous =
      match header idx moves.length fresh depth cut (schedule.available i q) (schedule.busy i q) word, moves[idx]? with
      | .split, _ => ⟨[],q,.handoff s,previous⟩
      | .postpone next bound, some _ =>
        walk schedule source depth cut (i+1) next bound ((MoveList.Ordering.postponeMove moves idx).drop next) s previous
      | .child next _ _ score, some m =>
        let update := after s.alpha s.best score
        let chosen := if update.2.1 then some m else previous
        if update.2.2 then ⟨[(m,word)],slots next fresh (moves.drop next),.returned score,chosen⟩
        else
          let tail := walk schedule source depth cut (i+1) next fresh (moves.drop next) ⟨s.alpha,update.1⟩ chosen
          {tail with calls := (m,word)::tail.calls}
      | _, _ => ⟨[],q,.returned s.best,previous⟩ := by
  dsimp only
  by_cases inside : idx < moves.length
  · let rest := moves.drop (idx+1)
    have rotated : (MoveList.Ordering.postponeMove moves idx).drop idx = rest ++ [moves[idx]] := by
      rw [MoveList.Ordering.postponeMove_eq _ _ inside,List.append_assoc,List.drop_append]
      simp [rest,Nat.le_of_lt inside]
    rw [walk.eq_def,List.getElem?_eq_getElem inside,List.drop_eq_getElem_cons inside]
    have size : idx + (moves[idx] :: rest).length = moves.length := by
      simp [rest]; omega
    dsimp only [Option.map,Option.getD]
    simp only [header,inside,ite_true,shouldPostpone,show moves.drop (idx+1) = rest from rfl,size]
    split
    · rfl
    · split
      · simp only [rotated]
      · rfl
  · rw [List.drop_eq_nil_of_le (by omega),List.getElem?_eq_none (by omega)]
    simp [walk,header,inside]

theorem covers (schedule : Schedule Move) (source : Nat → Move → Int) (depth : Nat) (cut : Bool)
    (i idx fresh : Nat) (pending : List Move) (s : State) (previous : Option Move) :
    let t := walk schedule source depth cut i idx fresh pending s previous
    (t.calls.map Prod.fst ++ t.remaining.pending).Perm pending := by
  rw [walk.eq_def]
  cases pending with
  | nil => simp
  | cons m rest =>
    dsimp only
    split
    · simp
    · split
      next rotate =>
        exact (covers schedule source depth cut (i+1) idx (fresh-1) (rest++[m]) s previous).trans
          (List.perm_append_singleton _ _)
      next =>
        split
        · simp
        · exact .cons _ (covers schedule source depth cut (i+1) (idx+1) fresh rest _ _)
termination_by pending.length + (fresh-idx)
decreasing_by
  · simp only [shouldPostpone,Bool.and_eq_true,decide_eq_true_eq] at *
    simp only [List.length_append,List.length_cons,List.length_nil]; omega
  · simp only [List.length_cons]; omega

theorem replays (schedule : Schedule Move) (source : Nat → Move → Int) (depth : Nat) (cut : Bool)
    (i idx fresh : Nat) (pending : List Move) (s : State) (previous : Option Move)
    (search : PvsMoveLoop.ChildSearch Move) (fuel : Nat)
    (fits : fresh-idx ≤ pending.length) (enough : (slots idx fresh pending).budget < fuel)
    (words : ∀ c ∈ (walk schedule source depth cut i idx fresh pending s previous).calls,
      search c.1 false (-(s.alpha+1)) (-s.alpha) = c.2) :
    run schedule depth cut search fuel i (slots idx fresh pending) s previous =
      some (walk schedule source depth cut i idx fresh pending s previous).receipt := by
  cases fuel with
  | zero => omega
  | succ fuel =>
    rw [run,walk.eq_def]
    rw [walk.eq_def] at words
    cases pending with
    | nil => simp [Trace.receipt]
    | cons m rest =>
      dsimp only at words ⊢
      simp only [pending_slots,List.isEmpty_cons,Bool.false_eq_true,ite_false]
      change (if canSplit idx (idx+(m::rest).length) depth cut _ then _ else _) = _
      split
      next hand =>
        rfl
      next hand =>
        simp only [hand,Bool.false_eq_true,ite_false] at words ⊢
        split at words
        next rotate =>
          have stepEq := step_rotate idx fresh m rest fits rotate
          have less := ((PvsPostpone.stepContract_proven Move).2.2 _ _ _ _ stepEq).2.1
          rw [dite_eq_left rotate,stepEq]
          apply replays _ _ _ _ _ _ _ _ _ _ _ fuel _ (by omega) words
          simp only [List.length_cons,List.length_append,List.length_nil] at fits ⊢
          omega
        next take =>
          have stepEq := step_take idx fresh m rest fits (Bool.eq_false_iff.mpr take)
          have less := ((PvsPostpone.stepContract_proven Move).2.2 _ _ _ _ stepEq).2.1
          rw [dite_eq_right take,stepEq]
          dsimp only [bind,Option.bind]
          split at words
          next stop =>
            have word := words (m,source i m) List.mem_cons_self
            dsimp only at word
            rw [word]
            simp [stop,Trace.receipt]
          next keep =>
            have word := words (m,source i m) List.mem_cons_self
            dsimp only at word
            rw [word]
            simp only [keep,ite_false,Bool.false_eq_true]
            rw [replays schedule source depth cut (i+1) (idx+1) fresh rest ⟨s.alpha,(after s.alpha s.best (-source i m)).1⟩
              (if (after s.alpha s.best (-source i m)).2.1 then some m else previous) search fuel (by
              simp only [List.length_cons] at fits; omega) (by omega)
              (fun c mem => words c (List.mem_cons_of_mem _ mem))]
            rfl
termination_by fuel

def start (schedule : Schedule Move) (source : Nat → Move → Int) (depth : Nat) (cut : Bool)
    (ordered : List Move) (idx : Nat) (alpha best : Int) :=
  walk schedule source depth cut 0 idx ordered.length (ordered.drop idx) ⟨alpha,best⟩ none

def Trace.value (t : Trace Move) (split : SplitReturn Move) (ordered : List Move) (idx : Nat) : Option Int :=
  match t.outcome with
  | .returned score => some score
  | .handoff _ => split ordered idx t.receipt

theorem recorded (schedule : Schedule Move) (source : Nat → Move → Int) (depth : Nat) (cut : Bool)
    (ordered : List Move) (idx : Nat) (alpha best : Int) (reply : Move → Option Int)
    (search : PvsMoveLoop.ChildSearch Move) (split : SplitReturn Move)
    (same : ∀ m, search m false (-(alpha+1)) (-alpha) = (reply m).getD 0)
    (words : ∀ c ∈ (start schedule source depth cut ordered idx alpha best).calls, reply c.1 = some c.2) :
    recordedOwner schedule depth cut search reply split alpha ordered idx best =
      (start schedule source depth cut ordered idx alpha best).value split ordered idx := by
  let t := start schedule source depth cut ordered idx alpha best
  let q : Queue Move := ⟨idx,ordered.drop idx,[]⟩
  have initial : slots idx ordered.length (ordered.drop idx) = q := by
    simp [slots,q,← List.length_drop]
    simp only [List.length_drop]; omega
  have fit : ordered.length-idx ≤ (ordered.drop idx).length := by simp
  have enough : (slots idx ordered.length (ordered.drop idx)).budget < q.budget+1 := by rw [initial]; omega
  have exec := replays schedule source depth cut 0 idx ordered.length (ordered.drop idx) ⟨alpha,best⟩ none
    search (q.budget+1) fit enough (fun c mem => by rw [same,words c mem]; rfl)
  rw [initial] at exec
  change run _ _ _ _ _ _ _ _ _ = some t.receipt at exec
  have allSeen : t.receipt.visited.all (fun m => (reply m).isSome) = true := by
    apply List.all_eq_true.mpr
    intro m member
    obtain ⟨c,mem,rfl⟩ := List.mem_map.mp member
    simp [words c mem]
  unfold recordedOwner
  change ((run _ _ _ _ _ _ q _ _).bind _) = t.value _ _ _
  rw [exec]
  simp only [Option.bind_some,allSeen,ite_true]
  cases h : t.outcome <;> simp [Trace.value,Trace.receipt,h]

structure Result (Move : Type) where
  calls : List (Move × Int)
  value : Option Int

def body (schedule : Schedule Move) (source : Nat → Move → Int) (depth : Nat) (cut : Bool)
    (ordered : List Move) (first : Move) (count : Nat) (hint : Option Move) (alpha beta pre : Int)
    (split : SplitReturn Move) : Result Move :=
  let loop := fun idx best =>
    let t := start schedule source depth cut ordered idx alpha best
    Result.mk t.calls (t.value split ordered idx)
  if count = 1 || hint.isSome then
    let m := hint.getD first
    if alpha < -pre && beta ≤ -pre then ⟨[(m,pre)],some (-pre)⟩
    else
      let tail := loop 1 (-pre)
      ⟨(m,pre)::tail.calls,tail.value⟩
  else loop 0 (-32767)

private theorem walk_words (schedule : Schedule Move) (source : Nat → Move → Int) (depth : Nat) (cut : Bool)
    (i idx fresh : Nat) (pending : List Move) (s : State) (previous : Option Move)
    (pred : Int → Prop) (raw : ∀ j m, pred (source j m)) :
    ∀ c ∈ (walk schedule source depth cut i idx fresh pending s previous).calls, pred c.2 := by
  rw [walk.eq_def]
  cases pending with
  | nil => simp
  | cons m rest =>
    dsimp only
    split
    · simp
    · split
      · exact walk_words _ _ _ _ _ _ _ _ _ _ pred raw
      · split
        · intro c mem
          exact List.mem_singleton.mp mem ▸ raw i m
        · intro c mem
          rcases List.mem_cons.mp mem with rfl | later
          · exact raw i m
          · exact walk_words _ _ _ _ _ _ _ _ _ _ pred raw c later
termination_by pending.length + (fresh-idx)
decreasing_by
  · simp only [shouldPostpone,Bool.and_eq_true,decide_eq_true_eq] at *
    simp only [List.length_append,List.length_cons,List.length_nil]; omega
  · simp only [List.length_cons]; omega

theorem body_words (schedule : Schedule Move) (source : Nat → Move → Int) (depth : Nat) (cut : Bool)
    (ordered : List Move) (first : Move) (count : Nat) (hint : Option Move) (alpha beta pre : Int)
    (split : SplitReturn Move) (pred : Int → Prop) (raw : ∀ j m, pred (source j m)) (preWord : pred pre) :
    ∀ c ∈ (body schedule source depth cut ordered first count hint alpha beta pre split).calls, pred c.2 := by
  unfold body
  split
  · split
    · intro c mem; exact List.mem_singleton.mp mem ▸ preWord
    · intro c mem
      rcases List.mem_cons.mp mem with rfl | later
      · exact preWord
      · exact walk_words _ _ _ _ _ _ _ _ _ _ pred raw c later
  · exact walk_words _ _ _ _ _ _ _ _ _ _ pred raw

theorem body_distinct (schedule : Schedule Move) (source : Nat → Move → Int) (depth : Nat) (cut : Bool)
    (ordered : List Move) (first : Move) (count : Nat) (hint : Option Move) (alpha beta pre : Int)
    (split : SplitReturn Move) (unique : ordered.Nodup)
    (head : count = 1 ∨ hint.isSome → ordered.head? = some (hint.getD first)) :
    ((body schedule source depth cut ordered first count hint alpha beta pre split).calls.map Prod.fst).Nodup := by
  have perm (idx best) := covers schedule source depth cut 0 idx ordered.length (ordered.drop idx) ⟨alpha,best⟩ none
  unfold body
  split
  next issued =>
    split
    · simp
    · have firstAt := head (by simpa using issued)
      obtain ⟨rest,atHead⟩ := List.head?_eq_some_iff.mp firstAt
      have tail := (perm 1 (-pre)).cons (hint.getD first)
      have full : hint.getD first :: ordered.drop 1 = ordered := by simp [atHead]
      rw [full] at tail
      exact (List.pairwise_append.mp (by simpa [start] using (tail.nodup_iff.mpr unique) :
        ((hint.getD first :: (start schedule source depth cut ordered 1 alpha (-pre)).calls.map Prod.fst) ++
          (start schedule source depth cut ordered 1 alpha (-pre)).remaining.pending).Nodup)).1
  next =>
    exact (List.pairwise_append.mp ((perm 0 (-32767)).nodup_iff.mpr (by simpa using unique))).1

theorem body_replays (schedule : Schedule Move) (source : Nat → Move → Int) (depth : Nat) (cut : Bool)
    (ordered : List Move) (first : Move) (count : Nat) (hint : Option Move) (alpha beta pre : Int)
    (split : SplitReturn Move) (search : PvsMoveLoop.ChildSearch Move) (reply : Move → Option Int)
    (same : ∀ m, search m false (-(alpha+1)) (-alpha) = (reply m).getD 0)
    (words : ∀ c ∈ (body schedule source depth cut ordered first count hint alpha beta pre split).calls,
      reply c.1 = some c.2) :
    (do
      if count = 1 || hint.isSome then let _ ← reply (hint.getD first)
      GeneralNonPvMir.resumeOwner (recordedOwner schedule depth cut search reply split alpha ordered)
        (GeneralNonPvMir.presearch count hint first alpha beta (fun m => search m false))) =
      (body schedule source depth cut ordered first count hint alpha beta pre split).value := by
  unfold body at words ⊢
  simp only [GeneralNonPvMir.presearch]
  by_cases issued : (decide (count = 1) || hint.isSome) = true
  · simp only [issued,ite_true] at words ⊢
    by_cases stop : (decide (alpha < -pre) && decide (beta ≤ -pre)) = true
    · simp only [stop,ite_true] at words
      have seen := words (hint.getD first,pre) List.mem_cons_self
      dsimp only at seen
      simp [seen,same,stop,GeneralNonPvMir.resumeOwner]
    · simp only [stop] at words
      have seen := words (hint.getD first,pre) List.mem_cons_self
      dsimp only at seen
      simp only [seen,same,Option.getD_some,stop,bind,Option.bind,GeneralNonPvMir.resumeOwner]
      exact recorded schedule source depth cut ordered 1 alpha (-pre) reply search split same
        (fun c mem => words c (List.mem_cons_of_mem _ mem))
  · simp only [issued,GeneralNonPvMir.resumeOwner] at words ⊢
    exact recorded schedule source depth cut ordered 0 alpha (-32767) reply search split same words

end NeuralReversi.PvsParallel.GeneralOwnerMir.Calls
