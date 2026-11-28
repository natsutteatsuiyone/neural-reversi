module
public import NeuralReversi.TranspositionTable.SeqLock

@[expose] public section

namespace NeuralReversi.TranspositionTable.WriterTrace

open SeqLock

structure Call (Event : Type) where
  expected : Word
  payload : Payload
  cas : Event
  fence : Event
  store : Field → Event
  finish : Event

inductive Action (Event : Type) where
  | claim (call : Call Event)
  | publish (call : Call Event)

def Action.word : Action Event → Word
  | .claim call => call.expected ||| 1
  | .publish call => SeqLock.publish call.expected

def Action.event : Action Event → Event
  | .claim call => call.cas
  | .publish call => call.finish

inductive Ordered (baseEpoch : Nat) : List (Action Event) → Nat → Option (Call Event) → Prop
  | initial : Ordered baseEpoch [] 0 none
  | claim : Ordered baseEpoch history epoch none → call.expected = seqWord (2 * baseEpoch + 2 * epoch) →
      Ordered baseEpoch (history ++ [.claim call]) epoch (some call)
  | publish : Ordered baseEpoch history epoch (some call) →
      Ordered baseEpoch (history ++ [.publish call]) (epoch + 1) none

private theorem claimWord (epoch : Nat) : seqWord (2 * epoch) ||| 1 = seqWord (2 * epoch + 1) := by
  have even := (seqWord_even_iff (2 * epoch)).mpr (by omega)
  rw [← BitVec.add_eq_or_of_and_eq_zero _ _ even]
  apply BitVec.eq_of_toNat_eq
  simp [seqWord, BitVec.toNat_add, Nat.add_mod]

private theorem Ordered.ownerExpected (trace : Ordered baseEpoch history epoch (some call)) :
    call.expected = seqWord (2 * baseEpoch + 2 * epoch) := by
  cases trace with
  | claim _ expected => exact expected

theorem Ordered.length_eq (trace : Ordered baseEpoch history epoch owner) :
    history.length = 2 * epoch + if owner.isSome then 1 else 0 := by
  induction trace with
  | initial => rfl
  | claim _ _ ih => simp at ih ⊢; omega
  | publish _ ih => simp at ih ⊢; omega

private theorem lastAction (history : List (Action Event)) (entry : Action Event) :
    (history ++ [entry])[history.length]? = some entry := by
  simp

private def completedActions : List (Call Event) → List (Action Event)
  | [] => []
  | call :: rest => .claim call :: .publish call :: completedActions rest

private theorem completedActions_append (closed : List (Call Event)) (call : Call Event) :
    completedActions (closed ++ [call]) = completedActions closed ++ [.claim call, .publish call] := by
  induction closed with
  | nil => rfl
  | cons _ _ ih => simp [completedActions, ih]

private theorem completedActions_length (closed : List (Call Event)) :
    (completedActions closed).length = 2 * closed.length := by
  induction closed with
  | nil => rfl
  | cons _ _ ih => simp [completedActions, ih]; omega

private theorem completedActions_at {closed : List (Call Event)} {i : Nat} {call : Call Event}
    (atIndex : closed[i]? = some call) :
    (completedActions closed)[2 * i]? = some (.claim call) ∧
    (completedActions closed)[2 * i + 1]? = some (.publish call) := by
  induction closed generalizing i with
  | nil => simp at atIndex
  | cons head tail ih =>
    cases i with
    | zero =>
      have same : head = call := Option.some.inj atIndex
      subst head
      exact ⟨rfl, rfl⟩
    | succ j =>
      have pair := ih (show tail[j]? = some call from atIndex)
      rw [show 2 * (j + 1) = 2 * j + 2 by omega]
      simpa [completedActions, Nat.add_assoc] using pair

private theorem Ordered.shape {history : List (Action Event)} {owner : Option (Call Event)}
    (trace : Ordered baseEpoch history epoch owner) :
    ∃ closed : List (Call Event), closed.length = epoch ∧
      history = completedActions closed ++ owner.toList.map Action.claim := by
  induction trace with
  | initial => exact ⟨[], rfl, rfl⟩
  | claim _ _ ih =>
    obtain ⟨closed, count, rfl⟩ := ih
    exact ⟨closed, count, by simp⟩
  | @publish prior epoch call _ ih =>
    obtain ⟨closed, count, rfl⟩ := ih
    exact ⟨closed ++ [call], by simp [count], by simp [completedActions_append]⟩

private theorem Ordered.claimAt {history : List (Action Event)} {owner : Option (Call Event)}
    (ordered : Ordered baseEpoch history count owner) (epoch : Nat) (positive : 0 < epoch)
    (begun : 2 * epoch - 1 ≤ history.length) :
    ∃ call, history[2 * epoch - 2]? = some (.claim call) ∧
      (2 * epoch ≤ history.length → history[2 * epoch - 1]? = some (.publish call)) := by
  obtain ⟨closed, closedCount, rfl⟩ := ordered.shape
  have length := completedActions_length closed
  by_cases finished : epoch ≤ count
  · have inside : epoch - 1 < closed.length := by omega
    let call := closed[epoch - 1]
    have atIndex : closed[epoch - 1]? = some call := List.getElem?_eq_getElem inside
    have pair := completedActions_at (call := call) atIndex
    have i : 2 * epoch - 2 = 2 * (epoch - 1) := by omega
    have j : 2 * epoch - 1 = 2 * (epoch - 1) + 1 := by omega
    refine ⟨call, ?_, fun _ => ?_⟩
    · rw [List.getElem?_append_left (by omega), i]
      exact pair.1
    · rw [List.getElem?_append_left (by omega), j]
      exact pair.2
  · cases owner with
    | none => simp [length] at begun; omega
    | some call =>
      simp [length] at begun
      have atEnd : 2 * epoch - 2 = (completedActions closed).length := by omega
      refine ⟨call, ?_, fun impossible => ?_⟩
      · rw [atEnd]
        exact lastAction _ _
      · simp [length] at impossible
        omega

private theorem Ordered.ownerOfClaim {history : List (Action Event)} {call : Call Event}
    (ordered : Ordered baseEpoch history epoch owner) (claimed : Action.claim call ∈ history)
    (unpublished : Action.publish call ∉ history) : owner = some call := by
  induction ordered with
  | initial => simp at claimed
  | claim _ _ ih =>
    rcases List.mem_append.mp claimed with earlier | last
    · have impossible := ih earlier (fun mem => unpublished (List.mem_append_left _ mem))
      contradiction
    · have same := Action.claim.inj (List.mem_singleton.mp last)
      rw [same]
  | @publish prior epoch previous _ ih =>
    have earlier : Action.claim call ∈ prior := by simpa using claimed
    have same := Option.some.inj (ih earlier (fun mem => unpublished (List.mem_append_left _ mem)))
    exact (unpublished (List.mem_append_right _ (by simp [same]))).elim

inductive ModificationOrder (baseEpoch : Nat) : List (Action Event) → Word → Prop
  | initial : ModificationOrder baseEpoch [] (seqWord (2 * baseEpoch))
  | claim (call : Call Event) : ModificationOrder baseEpoch history current → call.expected &&& 1 = 0 →
      (SeqLock.claim current call.expected).1 = true →
      ModificationOrder baseEpoch (history ++ [.claim call]) (.claim call : Action Event).word
  | publish (call : Call Event) : ModificationOrder baseEpoch history current → Action.claim call ∈ history →
      Action.publish call ∉ history →
      ModificationOrder baseEpoch (history ++ [.publish call]) (.publish call : Action Event).word

theorem ModificationOrder.initialOfEven (Event : Type) (value : Word)
    (even : value &&& 1 = 0) :
    ∃ baseEpoch, ModificationOrder (Event := Event) baseEpoch [] value := by
  have asWord : seqWord value.toNat = value := by simp [seqWord]
  have evenIndex := even_index_epoch value.toNat (by rwa [asWord])
  refine ⟨value.toNat / 2, ?_⟩
  have initial : seqWord (2 * (value.toNat / 2)) = value := by
    conv => lhs; rw [← evenIndex]
    exact asWord
  exact (congrArg (ModificationOrder (Event := Event) (value.toNat / 2) []) initial).mp .initial

theorem ModificationOrder.ordered (order : ModificationOrder baseEpoch history current) :
    ∃ epoch owner, Ordered baseEpoch history epoch owner ∧
      current = seqWord (2 * baseEpoch + history.length) := by
  induction order with
  | initial => exact ⟨0, none, .initial, rfl⟩
  | claim call _ even success ih =>
    obtain ⟨epoch, owner, ordered, value⟩ := ih
    have length := ordered.length_eq
    have accepted := (claim_success_iff _ _).mp success
    cases owner with
    | none =>
      simp at length
      have expected : call.expected = seqWord (2 * baseEpoch + 2 * epoch) := by
        rw [← accepted, value, length]
      exact ⟨epoch, some call, .claim ordered expected, by
        simpa [Action.word, expected, length, Nat.mul_add, Nat.add_assoc] using claimWord (baseEpoch + epoch)⟩
    | some _ =>
      simp at length
      have parity := (seqWord_even_iff (2 * baseEpoch + (2 * epoch + 1))).mp
        (by rwa [← length, ← value, accepted])
      omega
  | publish call _ prior fresh ih =>
    obtain ⟨epoch, _, ordered, _⟩ := ih
    have owned : Ordered baseEpoch _ epoch (some call) := ordered.ownerOfClaim prior fresh ▸ ordered
    have expected := owned.ownerExpected
    have length := owned.length_eq
    exact ⟨epoch + 1, none, .publish owned, by
      simp [Action.word, expected, publish_seqWord, length, Nat.add_assoc]⟩

private theorem wordAt_append {history : List (Action Event)} {last entry : Action Event} {i : Nat}
    (prior : history[i]? = some entry → entry.word = seqWord (2 * baseEpoch + (i + 1)))
    (value : last.word = seqWord (2 * baseEpoch + (history ++ [last]).length))
    (existsAt : (history ++ [last])[i]? = some entry) :
    entry.word = seqWord (2 * baseEpoch + (i + 1)) := by
  by_cases earlier : i < history.length
  · exact prior ((List.getElem?_append_left earlier).symm.trans existsAt)
  · obtain ⟨inside, _⟩ := List.getElem?_eq_some_iff.mp existsAt
    have atEnd : i = history.length := by simp at inside; omega
    subst i
    have same := Option.some.inj ((lastAction history last).symm.trans existsAt)
    simpa [← same] using value

theorem ModificationOrder.wordAt {history : List (Action Event)}
    (order : ModificationOrder baseEpoch history current) {i : Nat} {entry : Action Event}
    (existsAt : history[i]? = some entry) : entry.word = seqWord (2 * baseEpoch + (i + 1)) := by
  induction order with
  | initial => simp at existsAt
  | claim call reached even success ih =>
    obtain ⟨_, _, _, value⟩ := (ModificationOrder.claim call reached even success).ordered
    exact wordAt_append ih value existsAt
  | publish call reached prior fresh ih =>
    obtain ⟨_, _, _, value⟩ := (ModificationOrder.publish call reached prior fresh).ordered
    exact wordAt_append ih value existsAt

theorem ModificationOrder.callAt {history : List (Action Event)}
    (order : ModificationOrder baseEpoch history current) (epoch : Nat) (positive : 0 < epoch)
    (begun : 2 * epoch - 1 ≤ history.length) :
    ∃ call, history[2 * epoch - 2]? = some (.claim call) ∧
      (2 * epoch ≤ history.length → history[2 * epoch - 1]? = some (.publish call)) := by
  obtain ⟨_, _, ordered, _⟩ := order.ordered
  exact ordered.claimAt epoch positive begun

structure Sequence (memory : Memory Event) where
  history : List (Action Event)
  unique : (history.map Action.event).Nodup
  initialized : memory.writeValue .sequence 0 &&& 1 = 0
  source : ∀ i r, memory.rf (memory.write .sequence i) r → i ≤ history.length
  modified : ∀ i action, history[i]? = some action →
    memory.write .sequence (i + 1) = action.event ∧ memory.writeValue .sequence (i + 1) = action.word
  cas : ∀ i (call : Call Event), history[i]? = some (Action.claim call) →
    call.expected &&& 1 = 0 ∧ memory.acquire call.cas ∧
      memory.rf (memory.write .sequence i) call.cas ∧ memory.readValue call.cas = call.expected
  publish : ∀ (i : Nat) (call : Call Event), history[i]? = some (Action.publish call) →
    Action.claim call ∈ history ∧ memory.sb call.cas call.finish ∧ memory.release call.finish

def Sequence.baseEpoch {memory : Memory Event} (_seq : Sequence memory) : Nat :=
  (memory.writeValue .sequence 0).toNat / 2

theorem Sequence.initialWord {memory : Memory Event} (seq : Sequence memory) :
    memory.writeValue .sequence 0 = seqWord (2 * seq.baseEpoch) := by
  let raw := memory.writeValue .sequence 0
  have same : seqWord raw.toNat = raw := by simp [seqWord, raw]
  have even := even_index_epoch raw.toNat (by rw [same]; exact seq.initialized)
  change raw = seqWord (2 * (raw.toNat / 2))
  rw [← even, same]

private theorem Sequence.claimBefore {memory : Memory Event} (seq : Sequence memory)
    {i : Nat} (atFinish : seq.history[i]? = some (.publish call)) :
    Action.claim call ∈ seq.history.take i := by
  obtain ⟨present, localOrder, _⟩ := seq.publish i call atFinish
  obtain ⟨j, atClaim⟩ := List.mem_iff_getElem?.mp present
  have claim := (seq.modified j _ atClaim).1
  have finish := (seq.modified i _ atFinish).1
  have ordered := memory.sb_hb localOrder
  change memory.hb (Action.claim call).event (Action.publish call).event at ordered
  rw [← claim, ← finish] at ordered
  have lt := memory.write_write ordered
  exact List.mem_iff_getElem?.mpr ⟨j, by
    rw [List.getElem?_take]
    simp [show j < i by omega, atClaim]⟩

private theorem Sequence.freshFinish {memory : Memory Event} (seq : Sequence memory)
    {i : Nat} (atFinish : seq.history[i]? = some (.publish call)) :
    Action.publish call ∉ seq.history.take i := by
  intro repeated
  obtain ⟨j, inside, atEarlier⟩ := List.mem_iff_getElem.mp repeated
  have before : j < i := by simp at inside; omega
  have jIn : j < seq.history.length := by simp at inside; omega
  obtain ⟨iIn, current⟩ := List.getElem?_eq_some_iff.mp atFinish
  have previous : seq.history[j] = .publish call := by simpa only [List.getElem_take] using atEarlier
  have unique := List.pairwise_iff_getElem.mp seq.unique
  exact unique j i (by simpa using jIn) (by simpa using iIn) before
    (by simp [previous, current])

theorem Sequence.order {memory : Memory Event} (seq : Sequence memory) :
    ModificationOrder seq.baseEpoch seq.history (memory.writeValue .sequence seq.history.length) := by
  suffices ∀ n ≤ seq.history.length, ModificationOrder seq.baseEpoch (seq.history.take n)
      (memory.writeValue .sequence n) by simpa using this seq.history.length (Nat.le_refl _)
  intro n inside
  induction n with
  | zero => simpa only [List.take_zero, seq.initialWord] using
      (show ModificationOrder (Event := Event) seq.baseEpoch [] (seqWord (2 * seq.baseEpoch)) from .initial)
  | succ i ih =>
    have index : i < seq.history.length := inside
    have next : seq.history.take (i + 1) = seq.history.take i ++ [seq.history[i]] :=
      List.take_succ_eq_append_getElem index
    rw [next]
    have earlier := ih (Nat.le_of_lt index)
    have current : seq.history[i]? = some seq.history[i] := List.getElem?_eq_getElem index
    rw [(seq.modified i _ current).2]
    cases action : seq.history[i] with
    | claim call =>
      rw [action] at current
      obtain ⟨even, _, readsFrom, expected⟩ := seq.cas i call current
      exact .claim call earlier even ((claim_success_iff _ _).mpr
        ((memory.rf_value readsFrom).symm.trans expected))
    | publish call =>
      rw [action] at current
      exact .publish call earlier (seq.claimBefore current) (seq.freshFinish current)

/-- A final sequence load preceding the next writer's claim cannot read a
    later ordinal. The CAS reads the immediately preceding modification, so
    read-read coherence also bounds the relaxed validation load. -/
theorem Sequence.read_before_claim {memory : Memory Event} (seq : Sequence memory)
    (source cut : Nat) (load : Event) (observed : memory.rf (memory.write .sequence source) load)
    (next : cut < seq.history.length → ∃ c, seq.history[cut]? = some (.claim c) ∧ memory.hb load c.cas) :
    source ≤ cut := by
  by_cases inside : cut < seq.history.length
  · obtain ⟨c,atClaim,ahead⟩ := next inside
    exact memory.read_read ahead observed (seq.cas _ c atClaim).2.2.1
  · exact Nat.le_trans (seq.source _ _ observed) (by omega)

/-- A concrete release identifies both modifications by event identity, even if
    the sequence word has wrapped since an earlier writer. -/
theorem Sequence.publicationPair {memory : Memory Nat} (seq : Sequence memory)
    (published : Action.publish call ∈ seq.history) :
    let index := (seq.history.map Action.event).idxOf call.cas
    seq.history[index]? = some (.claim call) ∧ seq.history[index+1]? = some (.publish call) ∧
      memory.writeValue .sequence index = call.expected := by
  obtain ⟨i,atFinish⟩ := List.mem_iff_getElem?.mp published
  have inside := (List.getElem?_eq_some_iff.mp atFinish).1
  let k := i / 2
  obtain ⟨writer,atClaim,finished⟩ := seq.order.callAt (k+1) (by omega) (by dsimp [k]; omega)
  rw [show 2*(k+1)-2 = 2*k by omega] at atClaim
  have odd : i = 2*k+1 := by
    by_cases even : i = 2*k
    · rw [← even,atFinish] at atClaim
      cases atClaim
    · dsimp [k] at *
      omega
  have completed := finished (by omega)
  rw [show 2*(k+1)-1 = i by omega,atFinish] at completed
  cases completed
  have present : (seq.history.map Action.event)[2*k]? = some call.cas := by
    simp [atClaim,Action.event]
  obtain ⟨bound,atEvent⟩ := List.getElem?_eq_some_iff.mp present
  have index := seq.unique.idxOf_getElem (2*k) bound
  rw [atEvent] at index
  dsimp only
  rw [index,← odd]
  obtain ⟨_,_,source,value⟩ := seq.cas _ _ atClaim
  exact ⟨atClaim,atFinish,(memory.rf_value source).symm.trans value⟩

private theorem Sequence.claimEven {memory : Memory Event} (seq : Sequence memory)
    {i : Nat} (claimAt : seq.history[i]? = some (.claim call)) : i = 2 * (i / 2) := by
  have odd := claimed_is_odd call.expected
  have word := seq.order.wordAt claimAt
  change call.expected ||| 1 = _ at word
  rw [word] at odd
  have parity := mt (seqWord_even_iff _).mpr odd
  omega

private theorem Sequence.nextClaim {memory : Memory Event} (seq : Sequence memory)
    (epoch : Nat) (positive : 0 < epoch) (begun : 2 * epoch + 1 ≤ seq.history.length) :
    ∃ call, seq.history[2 * epoch]? = some (.claim call) ∧
      (2 * (epoch + 1) ≤ seq.history.length →
        seq.history[2 * epoch + 1]? = some (.publish call)) ∧
      memory.hb (memory.write .sequence (2 * epoch)) call.cas := by
  obtain ⟨call, claimAt, publishAt⟩ := seq.order.callAt (epoch + 1) (by omega) (by omega)
  rw [show 2 * (epoch + 1) - 2 = 2 * epoch by omega] at claimAt
  rw [show 2 * (epoch + 1) - 1 = 2 * epoch + 1 by omega] at publishAt
  obtain ⟨_, acquire, readsFrom, _⟩ := seq.cas (2 * epoch) call claimAt
  obtain ⟨previous, _, published⟩ := seq.order.callAt epoch positive (by omega)
  have done := published (by omega)
  have earlier := (seq.modified _ _ done).1
  rw [show 2 * epoch - 1 + 1 = 2 * epoch by omega] at earlier
  have release := (seq.publish _ previous done).2.2
  rw [earlier]
  exact ⟨call, claimAt, publishAt, memory.release_acquire release acquire (earlier ▸ readsFrom)⟩

private theorem Sequence.publicationsOrdered {memory : Memory Event} (seq : Sequence memory)
    (earlier later : Nat) (positive : 0 < earlier) (ordered : earlier < later)
    (finished : 2 * later ≤ seq.history.length) :
    memory.hb (memory.write .sequence (2 * earlier)) (memory.write .sequence (2 * later)) := by
  induction later with
  | zero => omega
  | succ epoch ih =>
    obtain ⟨call, _, published, acquired⟩ := seq.nextClaim epoch (by omega) (by omega)
    have done := published finished
    have physical := (seq.modified _ _ done).1
    rw [show 2 * epoch + 1 + 1 = 2 * (epoch + 1) by omega] at physical
    have next := memory.hb_trans acquired (memory.sb_hb (seq.publish _ call done).2.1)
    rw [physical]
    by_cases adjacent : earlier = epoch
    · simpa only [adjacent, Action.event] using next
    · exact memory.hb_trans (ih (by omega) (by omega)) next

private theorem Sequence.finishBeforeClaim {memory : Memory Event} (seq : Sequence memory)
    {earlier later : Nat} (published : seq.history[2 * earlier + 1]? = some (.publish first))
    (claimed : seq.history[2 * later]? = some (.claim next)) (ordered : earlier < later) :
    memory.hb first.finish next.cas := by
  have started := (List.getElem?_eq_some_iff.mp claimed).1
  obtain ⟨call, atClaim, _, acquired⟩ := seq.nextClaim later (by omega) (by omega)
  have same := Action.claim.inj (Option.some.inj (atClaim.symm.trans claimed))
  subst call
  have store := (seq.modified _ _ published).1
  rw [show 2 * earlier + 1 + 1 = 2 * (earlier + 1) by omega] at store
  change memory.write .sequence (2 * (earlier + 1)) = first.finish at store
  rw [← store]
  by_cases adjacent : earlier + 1 = later
  · simpa only [adjacent] using acquired
  · exact memory.hb_trans (seq.publicationsOrdered (earlier + 1) later (by omega) (by omega) (by omega)) acquired

private theorem Sequence.adjacentHB {memory : Memory Event} (seq : Sequence memory) (n : Nat)
    (present : n+2 ≤ seq.history.length) :
    memory.hb (memory.write .sequence (n+1)) (memory.write .sequence (n+2)) := by
  let k := n / 2
  by_cases even : n = 2*k
  · obtain ⟨call,claimed,published⟩ := seq.order.callAt (k+1) (by omega) (by omega)
    have done_ := published (by omega)
    rw [show 2*(k+1)-2 = n by omega] at claimed
    rw [show 2*(k+1)-1 = n+1 by omega] at done_
    rw [(seq.modified n _ claimed).1,(seq.modified (n+1) _ done_).1]
    exact memory.sb_hb (seq.publish _ _ done_).2.1
  · have odd : n+1 = 2*(k+1) := by dsimp [k] at *; omega
    obtain ⟨call,claimed,_,acquired⟩ := seq.nextClaim (k+1) (by omega) (by omega)
    rw [← odd] at claimed acquired
    rw [(seq.modified (n+1) _ claimed).1]
    exact acquired

private theorem Sequence.modificationsHB {memory : Memory Event} (seq : Sequence memory)
    (positive : 0 < i) (earlier : i < j) (present : j ≤ seq.history.length) :
    memory.hb (memory.write .sequence i) (memory.write .sequence j) := by
  induction j with
  | zero => omega
  | succ j ih =>
    obtain ⟨n,rfl⟩ := Nat.exists_eq_succ_of_ne_zero (show j ≠ 0 by omega)
    have next := seq.adjacentHB n present
    by_cases adjacent : i = n+1
    · exact adjacent ▸ next
    · exact memory.hb_trans (ih (by omega) (by omega)) next

private theorem Sequence.indexUnique {memory : Memory Event} (seq : Sequence memory)
    {i j : Nat} (left : seq.history[i]? = some action) (right : seq.history[j]? = some action) :
    i = j := by
  have inside := (List.getElem?_eq_some_iff.mp left).1
  exact seq.unique.eq_of_getElem?_eq (by simpa using inside) (by simp [left, right])

def fieldWord (payload : Payload) : Field → Word
  | .player => payload.player
  | .opponent => payload.opponent
  | .metadata => payload.metadata

structure FieldWrites {memory : Memory Event} (seq : Sequence memory) (field : Field) where
  history : List (Call Event)
  unique : history.Nodup
  source : ∀ i r, memory.rf (memory.write (.payload field) i) r → i ≤ history.length
  store : ∀ i call, history[i]? = some call →
    memory.write (.payload field) (i + 1) = call.store field ∧
    memory.writeValue (.payload field) (i + 1) = fieldWord call.payload field ∧
    Action.claim call ∈ seq.history ∧ memory.releaseFence call.fence ∧
    memory.sb call.cas call.fence ∧ memory.sb call.fence (call.store field)
  finish : ∀ call, Action.publish call ∈ seq.history →
    call ∈ history ∧ memory.sb (call.store field) call.finish

private theorem FieldWrites.storeHB {memory : Memory Event} {seq : Sequence memory} {f : Field}
    (field : FieldWrites seq f) {i j a b : Nat}
    (early : field.history[i]? = some first) (late : field.history[j]? = some next)
    (publication : seq.history[2 * a + 1]? = some (.publish first))
    (claim : seq.history[2 * b]? = some (.claim next)) (ordered : a < b) :
    memory.hb (memory.write (.payload f) (i+1)) (memory.write (.payload f) (j+1)) := by
  obtain ⟨event1, _, _, _, _, _⟩ := field.store i first early
  obtain ⟨event2, _, _, _, cas, write⟩ := field.store j next late
  have finish := (field.finish first (List.mem_iff_getElem?.mpr ⟨_, publication⟩)).2
  have between := seq.finishBeforeClaim publication claim ordered
  have hb := memory.hb_trans (memory.sb_hb finish)
    (memory.hb_trans between (memory.sb_hb (memory.sb_trans cas write)))
  rw [← event1, ← event2] at hb
  exact hb

private theorem FieldWrites.before {memory : Memory Event} {seq : Sequence memory} {f : Field}
    (field : FieldWrites seq f) {i j a b : Nat}
    (early : field.history[i]? = some first) (late : field.history[j]? = some next)
    (publication : seq.history[2 * a + 1]? = some (.publish first))
    (claim : seq.history[2 * b]? = some (.claim next)) (ordered : a < b) : i < j := by
  have coherent := memory.write_write (field.storeHB early late publication claim ordered)
  omega

theorem FieldWrites.aligned {memory : Memory Event} {seq : Sequence memory} {f : Field}
    (field : FieldWrites seq f) {i : Nat} (atWrite : field.history[i]? = some call) :
    seq.history[2 * i]? = some (.claim call) := by
  induction i using Nat.strongRecOn generalizing call with
  | ind i ih =>
    obtain ⟨_, _, member, _, _, _⟩ := field.store i call atWrite
    obtain ⟨j, atClaim⟩ := List.mem_iff_getElem?.mp member
    let rank := j / 2
    have index : j = 2 * rank := seq.claimEven atClaim
    rw [index] at atClaim
    by_cases earlier : rank < i
    · have inside : rank < field.history.length :=
        Nat.lt_trans earlier (List.getElem?_eq_some_iff.mp atWrite).1
      have found : field.history[rank]? = some field.history[rank] := List.getElem?_eq_getElem inside
      have same := Action.claim.inj (Option.some.inj ((ih rank earlier found).symm.trans atClaim))
      have equal := field.unique.eq_of_getElem?_eq inside
        (found.trans ((congrArg some same).trans atWrite.symm))
      omega
    · by_cases later : i < rank
      · have started := (List.getElem?_eq_some_iff.mp atClaim).1
        obtain ⟨first, firstClaim, published⟩ := seq.order.callAt (i + 1) (by omega) (by omega)
        have done := published (by omega)
        rw [show 2 * (i + 1) - 1 = 2 * i + 1 by omega] at done
        have present := (field.finish first (List.mem_iff_getElem?.mpr ⟨_, done⟩)).1
        obtain ⟨k, atEarlier⟩ := List.mem_iff_getElem?.mp present
        have before := field.before atEarlier atWrite done atClaim later
        have same := seq.indexUnique (ih k before atEarlier) firstClaim
        omega
      · have equal : i = rank := by omega
        simpa only [equal] using atClaim

structure Slot (memory : Memory Event) extends Sequence memory where
  fields : ∀ f, FieldWrites toSequence f

/-- Positive ranks up to this bound are precisely the recorded modifications of
    this atomic object, including stores by the current unpublished owner. -/
def Slot.modificationCount {memory : Memory Event} (slot : Slot memory) : Location → Nat
  | .sequence => slot.history.length
  | .payload f => (slot.fields f).history.length

theorem Slot.modificationsHB {memory : Memory Event} (slot : Slot memory) (loc : Location)
    (positive : 0 < i) (earlier : i < j) (present : j ≤ slot.modificationCount loc) :
    memory.hb (memory.write loc i) (memory.write loc j) := by
  cases loc with
  | sequence => exact slot.toSequence.modificationsHB positive earlier present
  | payload f =>
    obtain ⟨l,rfl⟩ := Nat.exists_eq_succ_of_ne_zero (show i ≠ 0 by omega)
    obtain ⟨r,rfl⟩ := Nat.exists_eq_succ_of_ne_zero (show j ≠ 0 by omega)
    let field := slot.fields f
    change r+1 ≤ field.history.length at present
    have left := List.getElem?_eq_getElem (show l < field.history.length by omega)
    have right := List.getElem?_eq_getElem (show r < field.history.length by omega)
    have claimed := field.aligned right
    have enough := (List.getElem?_eq_some_iff.mp claimed).1
    obtain ⟨call,first,completed⟩ := slot.toSequence.order.callAt (l+1) (by omega) (by omega)
    have published := completed (by omega)
    rw [show 2*(l+1)-2 = 2*l by omega] at first
    rw [show 2*(l+1)-1 = 2*l+1 by omega] at published
    have same := Action.claim.inj (Option.some.inj (first.symm.trans (field.aligned left)))
    subst call
    exact field.storeHB left right published claimed (by omega)

/-- A single numbering extending the happens-before edges across all slots
    already respects each local modification order of the seqlock implementation. -/
theorem linearizesModifications {Address Event : Type} (memory : Address → Memory Event)
    (slots : ∀ a, Slot (memory a)) (rank : Event → Nat)
    (extendsHB : ∀ a x y, (memory a).hb x y → rank x < rank y) (a : Address) (loc : Location)
    (positive : 0 < i) (earlier : i < j) (present : j ≤ (slots a).modificationCount loc) :
    rank ((memory a).write loc i) < rank ((memory a).write loc j) :=
  extendsHB a _ _ ((slots a).modificationsHB loc positive earlier present)

private theorem Slot.payloadRead {memory : Memory Event} (slot : Slot memory) (positive : 0 < k)
    (read : memory.rf (memory.write (.payload f) k) r) :
    ∃ call, (slot.fields f).history[k - 1]? = some call ∧ slot.history[2 * k - 2]? = some (.claim call) := by
  have bounded := (slot.fields f).source _ _ read
  have inside : k - 1 < (slot.fields f).history.length := by omega
  have atIndex := List.getElem?_eq_getElem inside
  refine ⟨_, atIndex, ?_⟩
  rw [show 2 * k - 2 = 2 * (k - 1) by omega]
  exact (slot.fields f).aligned atIndex

private theorem Slot.payloadSource {memory : Memory Event} (slot : Slot memory) (k : Nat) (f : Field) (r : Event)
    (positive : 0 < k) (read : memory.rf (memory.write (.payload f) k) r) :
    2 * k - 1 ≤ slot.history.length := by
  obtain ⟨_, _, claim⟩ := slot.payloadRead positive read
  have inside := (List.getElem?_eq_some_iff.mp claim).1
  omega

private theorem Slot.observedStore {memory : Memory Event} (slot : Slot memory) (k : Nat)
    (call : Call Event) (f : Field) (r : Event)
    (positive : 0 < k) (claim : slot.history[2 * k - 2]? = some (.claim call))
    (read : memory.rf (memory.write (.payload f) k) r) :
    memory.write (.payload f) k = call.store f ∧ memory.releaseFence call.fence ∧
      memory.sb call.cas call.fence ∧ memory.sb call.fence (call.store f) := by
  obtain ⟨actual, atIndex, atClaim⟩ := slot.payloadRead positive read
  have same := Action.claim.inj (Option.some.inj (atClaim.symm.trans claim))
  subst actual
  obtain ⟨write, _, _, fence, before, after⟩ := (slot.fields f).store _ _ atIndex
  rw [show k - 1 + 1 = k by omega] at write
  exact ⟨write, fence, before, after⟩

theorem FieldWrites.atPublication {memory : Memory Event} {seq : Sequence memory} {f : Field}
    (field : FieldWrites seq f) (positive : 0 < k)
    (done : seq.history[2*k-1]? = some (.publish call)) :
    field.history[k-1]? = some call := by
  have inside := (List.getElem?_eq_some_iff.mp done).1
  obtain ⟨actual, claimed, published⟩ := seq.order.callAt k positive (by omega)
  have eq := Action.publish.inj (Option.some.inj ((published (by omega)).symm.trans done))
  subst actual
  obtain ⟨member,_⟩ := field.finish _ (List.mem_of_getElem? done)
  obtain ⟨i, atIndex⟩ := List.mem_iff_getElem?.mp member
  have equal := seq.indexUnique (field.aligned atIndex) claimed
  rwa [show k-1 = i by omega]

private theorem Slot.finishedField {memory : Memory Event} (slot : Slot memory) (k : Nat)
    (call : Call Event) (positive : 0 < k) (done : slot.history[2 * k - 1]? = some (.publish call))
    (f : Field) :
    memory.write (.payload f) k = call.store f ∧
      memory.writeValue (.payload f) k = fieldWord call.payload f ∧
      memory.sb call.cas (call.store f) ∧ memory.sb (call.store f) call.finish := by
  have atIndex := (slot.fields f).atPublication positive done
  obtain ⟨write, value, _, _, before, after⟩ := (slot.fields f).store _ _ atIndex
  rw [show k-1+1 = k by omega] at write value
  exact ⟨write, value, memory.sb_trans before after, ((slot.fields f).finish _ (List.mem_of_getElem? done)).2⟩

def Slot.payloadAt {memory : Memory Event} (_slot : Slot memory) (k : Nat) : Payload :=
  ⟨memory.writeValue (.payload .player) k, memory.writeValue (.payload .opponent) k,
    memory.writeValue (.payload .metadata) k⟩

theorem Slot.publishedPayload {memory : Memory Event} (slot : Slot memory) (k : Nat)
    (positive : 0 < k) (published : 2 * k ≤ slot.history.length) :
    ∃ call, slot.history[2 * k - 1]? = some (.publish call) ∧ slot.payloadAt k = call.payload := by
  obtain ⟨call, _, publishAt⟩ := slot.toSequence.order.callAt k positive (by omega)
  have existsAt := publishAt published
  have player := (slot.finishedField k call positive existsAt .player).2.1
  have opponent := (slot.finishedField k call positive existsAt .opponent).2.1
  have metadata := (slot.finishedField k call positive existsAt .metadata).2.1
  exact ⟨call, existsAt, by simp [Slot.payloadAt, player, opponent, metadata, fieldWord]⟩

def Slot.fenceAt {memory : Memory Event} (slot : Slot memory) (epoch : Nat) : Event :=
  match slot.history[2 * epoch - 2]? with
  | some (.claim call) => call.fence
  | _ => memory.write .sequence 0

def Slot.writers {memory : Memory Event} (slot : Slot memory) : Writers memory where
  baseEpoch := slot.toSequence.baseEpoch
  begun k := 0 < k ∧ 2 * k - 1 ≤ slot.history.length
  published k := 0 < k ∧ 2 * k ≤ slot.history.length
  fence := slot.fenceAt
  sequence_value := by
    intro i r readsFrom
    cases i with
    | zero => simpa only [Nat.add_zero] using slot.toSequence.initialWord
    | succ n =>
      have inside : n < slot.history.length := slot.source _ _ readsFrom
      have actual := List.getElem?_eq_getElem inside
      exact (slot.modified _ _ actual).2.trans (slot.toSequence.order.wordAt actual)
  publication_read := by
    intro k r positive readsFrom
    exact ⟨positive, slot.source _ _ readsFrom⟩
  payload_read := by
    intro k f r positive readsFrom
    exact ⟨positive, slot.payloadSource _ _ _ positive readsFrom⟩
  publish_release := by
    intro k ⟨positive, done⟩
    obtain ⟨call, _, publishAt⟩ := slot.toSequence.order.callAt k positive (by omega)
    have actual := publishAt done
    have atIndex := (slot.modified _ _ actual).1
    rw [show 2 * k - 1 + 1 = 2 * k by omega] at atIndex
    rw [atIndex]
    exact (slot.publish _ _ actual).2.2
  payload_before_publish := by
    intro k f ⟨positive, done⟩
    obtain ⟨call, _, publishAt⟩ := slot.toSequence.order.callAt k positive (by omega)
    have actual := publishAt done
    have atIndex := (slot.modified _ _ actual).1
    rw [show 2 * k - 1 + 1 = 2 * k by omega] at atIndex
    obtain ⟨store, _, _, ordered⟩ := slot.finishedField _ _ positive actual f
    rw [atIndex, store]
    exact ordered
  payload_fences := by
    intro k f r readsFrom ⟨positive, begun⟩
    obtain ⟨call, atIndex, _⟩ := slot.toSequence.order.callAt k positive begun
    obtain ⟨store, release, before, after⟩ := slot.observedStore _ _ _ _ positive atIndex readsFrom
    have claim := (slot.modified _ _ atIndex).1
    rw [show 2 * k - 2 + 1 = 2 * k - 1 by omega] at claim
    rw [claim, store]
    simpa [Slot.fenceAt, atIndex, Action.event] using And.intro release (And.intro before after)

private theorem Slot.extendPayloadOrder {memory : Memory Event} (slot : Slot memory)
    (earlier epoch : Nat) (positive : 0 < earlier) (ordered : earlier ≤ epoch)
    (published : 2 * epoch ≤ slot.history.length) (field : Field)
    (after : memory.hb (memory.write .sequence (2 * epoch)) target) :
    memory.hb (memory.write (.payload field) earlier) target := by
  have fieldBefore := slot.writers.payload_before_publish earlier field ⟨positive, by omega⟩
  by_cases adjacent : earlier = epoch
  · exact memory.hb_trans (memory.sb_hb fieldBefore) (adjacent.symm ▸ after)
  · have before := slot.toSequence.publicationsOrdered earlier epoch positive (by omega) published
    exact memory.hb_trans (memory.sb_hb fieldBefore) (memory.hb_trans before after)

def PayloadOrderContract : Prop :=
  ∀ (Event : Type) (memory : Memory Event) (slot : Slot memory) (earlier later : Nat) (f g : Field),
    0 < earlier → earlier < later →
    (2 * later ≤ slot.history.length ∨ ∃ r, memory.rf (memory.write (.payload g) later) r) →
    memory.hb (memory.write (.payload f) earlier) (memory.write (.payload g) later)

theorem payloadOrderContract_proven : PayloadOrderContract := by
  intro Event memory slot earlier later f g positive ordered present
  let epoch := later - 1
  have index : epoch + 1 = later := by dsimp [epoch]; omega
  have begun : 2 * epoch + 1 ≤ slot.history.length := by
    rcases present with published | ⟨read, seen⟩
    · omega
    · have bound := slot.payloadSource later g read (by omega) seen
      omega
  obtain ⟨call, atClaim, atPublish, acquired⟩ := slot.toSequence.nextClaim epoch (by dsimp [epoch]; omega) begun
  have written : memory.hb call.cas (memory.write (.payload g) later) := by
    rcases present with published | ⟨read, seen⟩
    · have done := atPublish (by omega)
      rw [show 2 * epoch + 1 = 2 * later - 1 by omega] at done
      obtain ⟨stored, _, before, _⟩ := slot.finishedField later call (by omega) done g
      rw [stored]
      exact memory.sb_hb before
    · rw [show 2 * epoch = 2 * later - 2 by omega] at atClaim
      obtain ⟨stored, _, before, after⟩ := slot.observedStore later call g read (by omega) atClaim seen
      rw [stored]
      exact memory.sb_hb (memory.sb_trans before after)
  exact slot.extendPayloadOrder earlier epoch positive (by dsimp [epoch]; omega)
    (by omega) f (memory.hb_trans acquired written)

def OrderContract : Prop :=
  ∀ (Event : Type) (history : List (Action Event)) (current : Word) (baseEpoch : Nat),
    ModificationOrder baseEpoch history current →
    current = seqWord (2 * baseEpoch + history.length) ∧
    (∀ i entry, history[i]? = some entry → entry.word = seqWord (2 * baseEpoch + (i + 1))) ∧
    ∀ epoch, 0 < epoch → 2 * epoch - 1 ≤ history.length →
      ∃ call, history[2 * epoch - 2]? = some (.claim call) ∧
        (2 * epoch ≤ history.length → history[2 * epoch - 1]? = some (.publish call))

theorem orderContract_proven : OrderContract := by
  intro _ _ _ _ order
  obtain ⟨_, _, _, current⟩ := order.ordered
  exact ⟨current, fun _ _ => order.wordAt, order.callAt⟩

def AtomicLayoutContract : Prop :=
  ∀ (Event : Type) (memory : Memory Event) (slot : Slot memory),
    ModificationOrder slot.toSequence.baseEpoch slot.history (memory.writeValue .sequence slot.history.length) ∧
    (∀ f i call, (slot.fields f).history[i]? = some call → slot.history[2 * i]? = some (.claim call)) ∧
    ∀ k, 0 < k → 2 * k ≤ slot.history.length →
      ∃ call, slot.history[2 * k - 1]? = some (.publish call) ∧ slot.payloadAt k = call.payload

theorem atomicLayoutContract_proven : AtomicLayoutContract := by
  intro _ _ slot
  exact ⟨slot.toSequence.order, fun f _ _ => (slot.fields f).aligned, slot.publishedPayload⟩

example (call : Call Nat) (start : call.expected = 18446744073709551614) :
    ModificationOrder 9223372036854775807 [.claim call, .publish call] 0 := by
  have openCall : ModificationOrder 9223372036854775807 [.claim call] (.claim call : Action Nat).word :=
    .claim call .initial (by rw [start]; decide) (by rw [start]; decide)
  have closed := ModificationOrder.publish call openCall (by simp) (by simp)
  simpa [Action.word, SeqLock.publish, start] using closed

end NeuralReversi.TranspositionTable.WriterTrace
