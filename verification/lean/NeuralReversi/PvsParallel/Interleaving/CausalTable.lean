module
public import NeuralReversi.PvsParallel.Interleaving.Table

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.CausalTable

open NativeTableWrites.Journal
open TranspositionTable
open PvsBoardTree (Position minimax)
open Aspiration (Window SoundResult)
open FrameTableReads (Address)

abbrev Values := Position → Bool → Int → Int → Int → Prop
abbrev Lookups := Position → EntryData → Prop
abbrev Rule := Values → Lookups → Values

def Correct (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (uses : Values) : Prop :=
  ∀ b pv lo hi v, Window lo hi → (pv = false → hi = lo+1) → uses b pv lo hi v → SoundResult (minimax mc fc b) lo hi v

def Preserves (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (rule : Rule) : Prop :=
  ∀ children reads, Correct mc fc children → (∀ b e, reads b e → Table.EntrySafe mc fc b e) →
    Correct mc fc (rule children reads)

/-- A score is decided before a frame publishes its TT write. A parent consumes
    it only after the corresponding return. Prefixes can contain an unfinished frame. -/
structure Frame where
  chosen : Nat
  returned : Option Nat
  board : Position
  pv : Bool
  alpha : Int
  beta : Int
  score : Int

inductive Input (chosen : Values) (b : Position) : EntryData → Prop where
  | searched (pv : Bool) (lo hi score : Score) (d : Depth) (mv : Option Bitboard.Square)
      (gen : BitVec 8) (ttPv : Bool) (window : Window lo.raw.toInt hi.raw.toInt)
      (narrow : pv = false → hi.raw.toInt = lo.raw.toInt+1)
      (result : chosen b pv lo.raw.toInt hi.raw.toInt score.raw.toInt) (legal : TableReads.LegalChoice b mv) :
      Input chosen b (Packing.newData score mv (classify pv score lo hi) d Cutoffs.exactSelectivity gen true ttPv)
  | weaker (e : EntryData) (other : ¬Cutoffs.ExactEndgame e) (hint : TableReads.LegalHint b e) : Input chosen b e

theorem Input.safe (arg : Input chosen b e) (valid : Correct mc fc chosen) : Table.Input mc fc b e := by
  cases arg with
  | searched pv lo hi score d mv gen tag window narrow picked legal =>
    exact .searched pv lo hi score d mv gen tag window (valid _ _ _ _ _ window narrow picked) legal
  | weaker e other legal => exact .weaker e other legal

def Frame.values (f : Frame) : Values :=
  fun board pv alpha beta score => f.board = board ∧ f.pv = pv ∧ f.alpha = alpha ∧ f.beta = beta ∧ f.score = score

def chosen (before : Nat → Nat → Prop) (frames : List Frame) (consumer : Nat) : Values :=
  fun b pv lo hi v => ∃ f ∈ frames, before f.chosen consumer ∧ f.values b pv lo hi v

def returned (before : Nat → Nat → Prop) (frames : List Frame) (consumer : Nat) : Values :=
  fun b pv lo hi v => ∃ f ∈ frames, ∃ r, f.returned = some r ∧ before r consumer ∧ f.values b pv lo hi v

def lookup (x : NativeTableWrites.Journal.Execution c endian rs) (init : ∀ a, Even (x.memory a)) (consumer : Nat) : Lookups :=
  fun b e => e.bound ≠ 0 ∧ ∃ a, Table.Seen x a (init a) consumer (SlotJournal.packed endian b e)

/-- The relation may leave unrelated events unordered. Only actual frame decisions,
    returns and completed writer instructions participate in the dependencies. -/
structure Prefix (x : NativeTableWrites.Journal.Execution c endian rs) (init : ∀ a, Even (x.memory a))
    (frames : List Frame) (before : Nat → Nat → Prop) (rule : Rule) : Prop where
  irrefl : ∀ t, ¬before t t
  trans : ∀ {i j k}, before i j → before j k → before i k
  observes : ∀ a i j, (x.memory a).hb i j → before i j
  left : ∀ f ∈ frames, ∀ r, f.returned = some r → before f.chosen r
  body : ∀ f ∈ frames, rule (returned before frames f.chosen) (lookup x init f.chosen)
      f.board f.pv f.alpha f.beta f.score
  store : ∀ k, Record.writer ⟨k,.finish⟩ ∈ rs → (c k).success = true →
      Table.Operands endian (Input (chosen before frames ((c k).event 5)))
        (Table.Seen x (c k).address (init _) ((c k).event 5)) (c k).origin

private def finishEvent (c : Calls) : Record → Option Nat
  | .writer ⟨k,.finish⟩ => some ((c k).event 5)
  | _ => none

theorem finiteInduction (events : List Nat) (before : Nat → Nat → Prop) (p : Nat → Prop)
    (irrefl : ∀ i, ¬before i i) (trans : ∀ {i j k}, before i j → before j k → before i k)
    (step : ∀ i ∈ events, (∀ j ∈ events, before j i → p j) → p i) : ∀ i ∈ events, p i := by
  classical
  let past (i : Nat) := events.filter (fun j => decide (before j i))
  have lt {j i : Nat} (member : j ∈ events) (h : before j i) : (past j).length < (past i).length := by
    have contained : (past j).Sublist (past i) := by
      change (events.filter _).Sublist (events.filter _)
      clear member past step
      induction events with
      | nil => exact .slnil
      | cons t _ ih =>
        by_cases first : before t j
        · have last := trans first h
          simpa [first,last] using ih.cons_cons t
        · by_cases last : before t i
          · simpa [first,last] using ih.cons t
          · simpa [first,last] using ih
    have diff : past j ≠ past i := by
      intro eq
      have used : j ∈ past i := by simp [past,member,h]
      rw [←eq] at used
      simp [past,irrefl j] at used
    exact Nat.lt_of_le_of_ne contained.length_le (fun eq => diff (contained.eq_of_length eq))
  have bySize (n : Nat) : ∀ i ∈ events, (past i).length = n → p i := by
    induction n using Nat.strongRecOn with
    | ind n ih =>
      intro i mem size
      exact step i mem (fun j earlier order => ih _ (size ▸ lt earlier order) j earlier rfl)
  exact fun i mem => bySize _ _ mem rfl

private def CertifiedAt (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (c : Calls) (endian : Bool)
    (rs : List Record) (frames : List Frame) (t : Nat) : Prop :=
  (∀ f ∈ frames, f.chosen = t → Window f.alpha f.beta → (f.pv = false → f.beta = f.alpha+1) →
    SoundResult (minimax mc fc f.board) f.alpha f.beta f.score) ∧
  ∀ k, Record.writer ⟨k,.finish⟩ ∈ rs → (c k).success = true → (c k).event 5 = t →
    Table.Safe mc fc endian ((c k).call endian).payload

theorem certified (x : NativeTableWrites.Journal.Execution c endian rs) (init : ∀ a, Even (x.memory a))
    (trace : Prefix x init frames before rule) (kernel : Preserves mc fc rule)
    (seed : ∀ a, Table.Safe mc fc endian ((x.slot a (init a)).payloadAt 0)) :
    (∀ f ∈ frames, Window f.alpha f.beta → (f.pv = false → f.beta = f.alpha+1) →
      SoundResult (minimax mc fc f.board) f.alpha f.beta f.score) ∧
    ∀ t b e, lookup x init t b e → Table.EntrySafe mc fc b e := by
  let events := frames.map Frame.chosen ++ rs.filterMap (finishEvent c)
  have writer {k} (mem : Record.writer ⟨k,.finish⟩ ∈ rs) : (c k).event 5 ∈ events :=
    List.mem_append_right _ (List.mem_filterMap.mpr ⟨_,mem,rfl⟩)
  have frame {f} (mem : f ∈ frames) : f.chosen ∈ events := List.mem_append_left _ (List.mem_map.mpr ⟨_,mem,rfl⟩)
  let prop := CertifiedAt mc fc c endian rs frames
  have safeWords (t : Nat) (prior : ∀ j ∈ events, before j t → prop j)
      (a : Address) (p : SeqLock.Payload) (seen : Table.Seen x a (init a) t p) : Table.Safe mc fc endian p := by
    cases seen with
    | initial => exact seed a
    | stored n k hit hb =>
      obtain ⟨mem,_,yes,_⟩ := mem_mods.mp (List.mem_of_getElem? hit)
      exact (prior _ (writer mem) (trace.observes _ _ _ hb)).2 k mem yes rfl
  have proof : ∀ i ∈ events, prop i := by
    apply finiteInduction events before prop trace.irrefl trace.trans
    intro t mem prior
    constructor
    · intro f inside eq window narrow
      subst t
      apply kernel _ _ _ _ f.board f.pv f.alpha f.beta f.score window narrow (trace.body f inside)
      · intro b pv lo hi v window narrow uses
        obtain ⟨old,issued,r,exit,earlier,same⟩ := uses
        obtain ⟨rfl,rfl,rfl,rfl,rfl⟩ := same
        exact (prior _ (frame issued) (trace.trans (trace.left old issued r exit) earlier)).1 old issued rfl window narrow
      · intro b e used
        obtain ⟨occupied,a,seen⟩ := used
        have safe := safeWords _ prior a _ seen
        simpa only [SlotJournal.packed,Packing.unpack_pack] using
          (safe b rfl (by simpa only [SlotJournal.packed,Packing.unpack_pack] using occupied))
    · intro k published success eq
      subst t
      have operands := trace.store k published success
      have known : Correct mc fc (chosen before frames ((c k).event 5)) := by
        intro b pv lo hi v window narrow uses
        obtain ⟨old,issued,earlier,same⟩ := uses
        obtain ⟨rfl,rfl,rfl,rfl,rfl⟩ := same
        exact (prior _ (frame issued) earlier).1 old issued rfl window narrow
      exact Table.Produced.safe (operands.mapInputs (fun _ _ h => h.safe known)) (fun _ seen => safeWords _ prior _ _ seen)
  constructor
  · exact fun f present => (proof _ (frame present)).1 f present rfl
  · intro t b e used
    obtain ⟨occupied,a,seen⟩ := used
    have safe := safeWords t (fun j present _ => proof j present) a _ seen
    simpa only [SlotJournal.packed,Packing.unpack_pack] using
      (safe b rfl (by simpa only [SlotJournal.packed,Packing.unpack_pack] using occupied))

end NeuralReversi.PvsParallel.Interleaving.CausalTable
