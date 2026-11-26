import NeuralReversi.PvsParallel.SplitRebookingMir.Ordering

namespace NeuralReversi.PvsParallel.SplitRebookingMir

open Threading
open AtomicWord (Event Write Read)
open CutoffVisibility (History HappensBefore)
open SplitCompletion (Worker Action MaskState step initial run)
open SplitActivationMir (Command command code)

def executeMask (old : BitVec 64) : List Command → List Write
  | [] => []
  | cmd :: rest =>
    let write := cmd.apply old
    write :: executeMask write.value rest

private theorem execute_tail (s : MaskState) (actions : List Action) :
    (executeMask s.bits (actions.map command)).map Write.value = updates .mask s actions := by
  induction actions generalizing s with
  | nil => rfl
  | cons act rest ih =>
    have word : ((command act).apply s.bits).value = (step s act).bits := by cases act <;> rfl
    change ((command act).apply s.bits).value :: _ = (step s act).bits :: _
    rw [word]; congr 1
    simpa only [word] using ih (step s act)

private def ofWords (writes : List Write) (field : Field) (s : MaskState) (xs : List Action)
    (before after : List (BitVec 64))
    (eq : writes.map Write.value = before ++ values field s xs ++ after) : Prefix writes field s xs := by
  have len := congrArg List.length eq
  have pos : 0 < (values field s xs).length := by simp [values]
  simp only [List.length_map,List.length_append] at len
  exact ⟨⟨before.length, by omega⟩,⟨before.length + (values field s xs).length - 1, by omega⟩,
    before,after,eq,rfl,by change before.length + (values field s xs).length - 1 + 1 = _; omega⟩

def maskPrefix (writes prior suffix : List Write) (owner : Worker) (clear mark : Event) (acts : List Action)
    (old : BitVec 64)
    (mo : writes = prior ++ executeMask old (code owner clear mark acts) ++ suffix) :
    Prefix writes .mask (initial owner) acts := by
  apply ofWords writes .mask (initial owner) acts (prior.map Write.value ++ [0]) (suffix.map Write.value)
  rw [mo,SplitActivationMir.code_eq]
  simp only [List.map_append,executeMask,List.map_cons,Command.apply]
  simpa [values,Field.word,initial,Bitboard.squareMask, List.append_assoc] using
    congrArg (fun words => prior.map Write.value ++ 0 :: ((1 : BitVec 64) <<< owner.val) ::
      (words ++ suffix.map Write.value)) (execute_tail (initial owner) acts)

def flagStore (e : Event) : SplitActivationMir.Op → Option Write
  | .accept => some ⟨e,1,false,false⟩
  | .close => some ⟨e,0,false,false⟩
  | _ => none

def flagWrites (ownerOpen : Event) (closeTime : Event → Event) (actions : List Action) : List Write :=
  SplitActivationMir.initialOps.filterMap (flagStore ownerOpen) ++ actions.flatMap fun
    | .book _ e => SplitActivationMir.enrollOps.filterMap (flagStore e)
    | .finish _ e => SplitActivationMir.releaseOps.filterMap (flagStore (closeTime e))

private theorem flag_tail (s : MaskState) (closeTime : Event → Event) (actions : List Action) :
    (actions.flatMap (fun a => match a with
      | .book _ e => SplitActivationMir.enrollOps.filterMap (flagStore e)
      | .finish _ e => SplitActivationMir.releaseOps.filterMap (flagStore (closeTime e)))).map Write.value =
      updates .accepting s actions := by
  induction actions generalizing s with
  | nil => rfl
  | cons a xs ih =>
    cases a with
    | book w e => exact ih (step s (.book w e))
    | finish w e =>
      have close : Field.accepting.word (step s (.finish w e)) = 0 := by simp [Field.word,step,Action.release]
      change 0 :: _ = (Field.accepting.word (step s (.finish w e))) :: _
      rw [close]; exact congrArg (List.cons _) (ih (step s (.finish w e)))

def flagPrefix (writes prior suffix : List Write) (owner : Worker) (opened : Event) (closeTime : Event → Event)
    (acts : List Action) (mo : writes = prior ++ flagWrites opened closeTime acts ++ suffix) :
    Prefix writes .accepting (initial owner) acts := by
  apply ofWords writes .accepting (initial owner) acts (prior.map Write.value) (suffix.map Write.value)
  rw [mo]
  simp only [List.map_append, flagWrites]
  rw [flag_tail (initial owner)]
  rfl

private theorem last_value (f : Field) (s : MaskState) (acts : List Action) :
    (values f s acts).getLast? = some (f.word (run s acts)) := by
  induction acts generalizing s with
  | nil => rfl
  | cons a xs ih =>
    cases f <;> cases a with
    | book w e =>
      simpa [values,updates,changes,step,Action.release,Field.word,run] using ih (step s (.book w e))
    | finish w e =>
      simpa [values,updates,changes,step,Action.release,Field.word,run] using ih (step s (.finish w e))

theorem Prefix.final {words : List Write} (p : Prefix words f s xs) : (words.get p.last).value = f.word (run s xs) := by
  let vs := values f s xs
  have pos : 0 < vs.length := by simp [vs,values]
  have endEq := p.ends
  have idx : p.last.val = p.before.length + (vs.length - 1) := by dsimp [vs] at *; omega
  have sample : (words.map Write.value)[p.last.val]? = some (words.get p.last).value := by simp
  rw [p.words,idx, List.getElem?_append_left (by simp [vs] at *; omega),
    List.getElem?_append_right (by omega),Nat.add_sub_cancel_left,← List.getLast?_eq_getElem?] at sample
  exact Option.some.inj (sample.symm.trans (last_value _ _ _))

theorem locked_count (history : History) (words : List Write) (p : Prefix words .mask s acts)
    (load : Read words) (check : SplitPublication.LockedFlagRead history load p.last p.last.val)
    (admitted : room late (AtomicBitSet.count (words.get load.source).value) = true) :
    room late (AtomicBitSet.count (run s acts).bits) = true := by
  have bounds := check.window.bounds
  have source : load.source = p.last := Fin.ext (by omega)
  simpa only [source,p.final,Field.word] using admitted

theorem locked_open (history : History) (words : List Write) (p : Prefix words .accepting s acts)
    (load : Read words) (check : SplitPublication.LockedFlagRead history load p.last p.last.val)
    (admitted : (words.get load.source).value = 1) : (run s acts).accepting = true := by
  have bounds := check.window.bounds
  have source : load.source = p.last := Fin.ext (by omega)
  rw [source,p.final] at admitted
  cases h : (run s acts).accepting <;> simp_all [Field.word]

end NeuralReversi.PvsParallel.SplitRebookingMir
