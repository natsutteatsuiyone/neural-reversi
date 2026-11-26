import NeuralReversi.PvsParallel.FrameTableReads.Operations

namespace NeuralReversi.PvsParallel.FrameTableReads.Operations

open TranspositionTable
open SeqLock (Word)
open ReaderMir (AtomicAction)
open FrameReturns (Log Frame)

abbrev Issued := Nat → Op → Prop

def Runs (issued : Issued) (start : Nat) (ops : List Op) : Prop :=
  ∀ i op, ops[i]? = some op → issued (start+i) op

theorem Runs.append (h : Runs issued start (xs ++ ys)) :
    Runs issued start xs ∧ Runs issued (start+xs.length) ys := by
  constructor
  · intro i op get
    exact h i op (by rw [List.getElem?_append_left ((List.getElem?_eq_some_iff.mp get).choose),get])
  · intro i op get
    have hit : (xs ++ ys)[xs.length+i]? = some op := by rw [List.getElem?_append_right (by omega)]; simpa
    simpa [Nat.add_assoc] using h _ _ hit

/-- Facts about individual executed instructions in one search frame. Different
    addresses have their own modification orders. The no-wrap hypothesis only
    relates the two sequence loads of the same occupied-size reader. -/
structure Machine {log : Log mc fc} (t : Table log) (consumer : Nat) (issued : Issued) (s : Supply) where
  event : Nat → Nat
  origin : Nat → Nat
  word : ∀ r i loc mode, issued i (.slot r (.load loc mode)) → s.word r.address i = (t.memory r.address).readValue (event i)
  reads : ∀ r i loc mode, issued i (.slot r (.load loc mode)) →
    (t.memory r.address).rf ((t.memory r.address).write loc (origin i)) (event i)
  acquire : ∀ r i loc, issued i (.slot r (.load loc .acquire)) → (t.memory r.address).acquire (event i)
  fence : ∀ r i, issued i (.slot r (.fence .acquire)) → (t.memory r.address).acquireFence (event i)
  adjacent : ∀ r i x y, issued i (.slot r x) → issued (i+1) (.slot r y) → (t.memory r.address).sb (event i) (event (i+1))
  allocated : ∀ r, issued r.opening (.slot r (.load .sequence .acquire)) → log.before (t.ready r.address) (event r.opening)
  consumed : ∀ r, issued (r.opening+5) (.slot r (.load .sequence .relaxed)) → (t.memory r.address).sb (event (r.opening+5)) consumer
  noWrap : ∀ r, issued r.opening (.slot r (.load .sequence .acquire)) →
    issued (r.opening+5) (.slot r (.load .sequence .relaxed)) → origin (r.opening+5) < origin r.opening + SeqLock.modulus

/-- The extracted reader consults a word only at indices where it actually
    issued a load. Unused registers, including the fence position, are arbitrary. -/
structure ReadPath (code : Code) (value : (Nat → Word) → α) : Prop where
  first : ∀ w, (code w)[0]? = some (.load .sequence .acquire)
  final : ∀ w, (code w).length = 6 → (code w)[5]? = some (.load .sequence .relaxed)
  congr : ∀ w v, (∀ i loc mode, (code w)[i]? = some (.load loc mode) → w i = v i) →
    code w = code v ∧ value w = value v

private theorem snapshot_congr (endian : Bool) (w v : Nat → Word)
    (same : ∀ i loc mode, (snapCode endian w)[i]? = some (.load loc mode) → w i = v i) :
    ReaderMir.run endian w = ReaderMir.run endian v := by
  unfold snapCode ReaderMir.run at same
  unfold ReaderMir.run
  repeat' first
    | split at same
    | assumption
    | contradiction
  all_goals
    have first := same 0 _ _ rfl
    first
    | solve | simp_all
    | have metadata := same 1 _ _ rfl
      first
      | have player := same 2 _ _ rfl
        have opponent := same 3 _ _ rfl
        have last := same 5 _ _ rfl
        simp_all
      | have last := same 3 _ _ rfl
        simp_all

theorem snapshot_path (endian : Bool) : ReadPath (snapCode endian) (snapshot endian) where
  first w := by unfold snapCode ReaderMir.run; (repeat' split) <;> rfl
  final w := by unfold snapCode ReaderMir.run; (repeat' split) <;> simp
  congr w v same := by simp [snapCode,snapshot,snapshot_congr endian w v same]

private theorem lookup_congr (endian : Bool) (b : PvsBoardTree.Position) (w v : Nat → Word)
    (same : ∀ i loc mode, (lookupCode endian b w)[i]? = some (.load loc mode) → w i = v i) :
    ReaderMir.lookupRun endian b.val.player b.val.opponent w = ReaderMir.lookupRun endian b.val.player b.val.opponent v := by
  unfold lookupCode ReaderMir.lookupRun at same
  unfold ReaderMir.lookupRun
  repeat' split at same
  all_goals
    have first := same 0 _ _ rfl
    first
    | solve | simp_all
    | have player := same 1 _ _ rfl
      have opponent := same 2 _ _ rfl
      first
      | have metadata := same 3 _ _ rfl
        have last := same 5 _ _ rfl
        simp_all
      | simp_all

theorem lookup_path (endian : Bool) (b : PvsBoardTree.Position) : ReadPath (lookupCode endian b) (lookupValue endian b) where
  first w := by unfold lookupCode ReaderMir.lookupRun; (repeat' split) <;> rfl
  final w := by unfold lookupCode ReaderMir.lookupRun; (repeat' split) <;> simp
  congr w v same := by simp [lookupCode,lookupValue,lookup_congr endian b w v same]

theorem Site.issued_at (r : Site) (ran : Runs issued r.opening (r.emit s code))
    (action : (code (r.words s))[i]? = some op) : issued (r.opening+i) (.slot r op) :=
  ran i _ (by simp [Site.emit,action])

noncomputable def Machine.invocation {log : Log mc fc} {t : Table log} (m : Machine t consumer issued s)
    (r : Site) (path : ReadPath code value) (ran : Runs issued r.opening (r.emit s code)) :
    InvocationAt t r.address consumer code := by
  let ev := fun i => m.event (r.opening+i)
  have words := path.congr (r.words s) ((t.memory r.address).readValue ∘ ev)
    (fun i loc mode hit => m.word r _ loc mode (r.issued_at ran hit))
  have first : issued r.opening (.slot r (.load .sequence .acquire)) := r.issued_at ran (path.first _)
  have final := fun size => r.issued_at ran (path.final _ size)
  refine ⟨ev,fun i => m.origin (r.opening+i),m.allocated r first,?_,?_,?_⟩
  · rw [←words.1]
    constructor
    · intro i bound
      have hi : i < (code (r.words s)).length := by omega
      apply m.adjacent r (r.opening+i) _ _ (r.issued_at ran (List.getElem?_eq_getElem hi))
      exact r.issued_at ran (List.getElem?_eq_getElem bound)
    · exact fun i loc mode hit => m.reads r _ _ _ (r.issued_at ran hit)
    · exact fun i loc hit => m.acquire r _ _ (r.issued_at ran hit)
    · exact fun i hit => m.fence r _ (r.issued_at ran hit)
  · intro size
    exact m.consumed r (final (words.1.symm ▸ size))
  · intro size
    exact m.noWrap r first (final (words.1.symm ▸ size))

theorem Machine.value {value : (Nat → Word) → α} {log : Log mc fc} {t : Table log} (m : Machine t consumer issued s)
    (r : Site) (path : ReadPath code value) (ran : Runs issued r.opening (r.emit s code)) :
    value (m.invocation r path ran).words = value (r.words s) :=
  (path.congr _ _ (fun _ loc mode hit => m.word r _ loc mode (r.issued_at ran hit))).2.symm

end NeuralReversi.PvsParallel.FrameTableReads.Operations
