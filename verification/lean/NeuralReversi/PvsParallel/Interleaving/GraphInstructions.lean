module
public import NeuralReversi.PvsParallel.Interleaving.JournalAtomicGraph

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.GraphInstructions

open TranspositionTable
open SeqLock (Word Location)
open FrameTableReads (Address)
open NativeTableWrites.Journal (Calls Record Site Pc mods)
open JournalAtomicGraph (Graph WriterOK)
open ModificationIndex (word decode)

def rank (g : Graph) (c : Calls) (rs : List Record) (n : Nat) (a : Address) (l : Location) (r : Nat) : Option Nat :=
  (g.source a r).bind fun w => (decode c rs n a w).bind fun (k,i) => if k = l then some i else none

private theorem rank_eq : rank g c rs n a l r = some i ↔
    ∃ w, g.source a r = some w ∧ decode c rs n a w = some (l,i) := by
  simp only [rank,Option.bind_eq_some_iff]
  constructor
  · rintro ⟨w,src,⟨⟨k,j⟩,found,picked⟩⟩
    split at picked <;> simp_all
  · rintro ⟨w,src,found⟩
    exact ⟨w,src,⟨(l,i),found,by simp⟩⟩

def checkWriter (g : Graph) (c : Calls) (endian : Bool) (rs : List Record) (n : Nat) (s : Site) : Bool :=
  let call := c s.invocation
  match s.pc with
  | .cas => (rank g c rs n call.address .sequence (call.event 0)).any fun i =>
      decide ((call.success = true ↔ word c endian rs g.initial call.address .sequence i = call.expected.val) ∧
        (call.success = true → (mods c rs call.address .sequence)[i]? = some s))
  | _ => call.success

private theorem writer_eq (s : Site) :
    checkWriter g c endian rs n s = true ↔ WriterOK g c endian rs n s := by
  obtain ⟨k,pc⟩ := s
  cases pc <;> try rfl
  simp only [checkWriter,WriterOK,Option.any_eq_true,rank_eq,decide_eq_true_eq]
  constructor
  · rintro ⟨i,⟨w,raw,site⟩,compared,next⟩
    exact ⟨w,i,raw,site,compared,next⟩
  · rintro ⟨w,i,raw,site,compared,next⟩
    exact ⟨i,⟨w,raw,site⟩,compared,next⟩

def checkLoad (g : Graph) (c : Calls) (endian : Bool) (rs : List Record) (es : Joint.Tape Op) (r : Nat) : Bool :=
  match Joint.opcode es r with
  | some (.load a l _) => (rank g c rs es.length a l r).any fun i =>
      decide (word c endian rs g.initial a l i = Joint.word es r)
  | _ => true

def LoadAt (g : Graph) (c : Calls) (endian : Bool) (rs : List Record) (es : Joint.Tape Op) (r : Nat) :=
  ∀ a l mode, Joint.opcode es r = some (.load a l mode) →
    ∃ w i, g.source a r = some w ∧ decode c rs es.length a w = some (l,i) ∧
      word c endian rs g.initial a l i = Joint.word es r

private theorem load_eq : checkLoad g c endian rs es r = true ↔ LoadAt g c endian rs es r := by
  cases op : Joint.opcode es r with
  | none => simp [LoadAt,checkLoad,op]
  | some cmd =>
    cases cmd <;> simp [LoadAt,checkLoad,op,Option.any_eq_true,rank_eq]
    constructor
    · rintro ⟨i,⟨w,raw,site⟩,eq⟩
      exact ⟨w,raw,i,site,eq⟩
    · rintro ⟨w,raw,i,site,eq⟩
      exact ⟨i,⟨w,raw,site⟩,eq⟩

def Writers (g : Graph) (c : Calls) (endian : Bool) (rs : List Record) (es : Joint.Tape Op) : Prop :=
  ∀ s, Record.writer s ∈ rs → WriterOK g c endian rs es.length s

def Loads (g : Graph) (c : Calls) (endian : Bool) (rs : List Record) (es : Joint.Tape Op) : Prop :=
  ∀ r, LoadAt g c endian rs es r

/-- Check each recorded writer PC and each issued slot-field load. A CAS uses the
    actual expected operand and, on success, must immediately follow its source in MO. -/
def check (g : Graph) (c : Calls) (endian : Bool) (rs : List Record) (es : Joint.Tape Op) : Bool :=
  rs.all (fun | .writer s => checkWriter g c endian rs es.length s | _ => true) &&
    (List.range es.length).all (checkLoad g c endian rs es)

theorem check_iff : check g c endian rs es = true ↔ Writers g c endian rs es ∧ Loads g c endian rs es := by
  simp only [check,Bool.and_eq_true,List.all_eq_true,List.mem_range,load_eq]
  constructor
  · rintro ⟨writes,reads⟩
    refine ⟨fun s mem => (writer_eq s).mp (writes (.writer s) mem),fun r => ?_⟩
    by_cases inside : r < es.length
    · exact reads r inside
    · have absent : Joint.opcode es r = none := by
        simp [Joint.opcode,List.getElem?_eq_none (Nat.le_of_not_gt inside)]
      simp [LoadAt,absent]
  · rintro ⟨writes,reads⟩
    refine ⟨fun row mem => ?_,fun r _ => reads r⟩
    cases row with
    | localCall => rfl
    | writer s => exact (writer_eq s).mpr (writes s mem)

theorem writers_of_check (h : check g c endian rs es = true) : Writers g c endian rs es := (check_iff.mp h).1
theorem loads_of_check (h : check g c endian rs es = true) : Loads g c endian rs es := (check_iff.mp h).2

def Contract : Prop := ∀ (Op : Type) (g : Graph) (c : Calls) (endian : Bool)
    (rs : List Record) (es : Joint.Tape Op), check g c endian rs es = true ↔ Writers g c endian rs es ∧ Loads g c endian rs es

theorem certified : Contract := fun _ _ _ _ _ _ => check_iff

namespace Regression

private def a : Address := ⟨1,0⟩
private def pcs : List Pc := [.cas,.fence,.field .player,.field .opponent,.field .metadata,.finish]
private def sites : List Record := pcs.map (fun pc => .writer ⟨0,pc⟩) ++ [.writer ⟨1,.cas⟩,.writer ⟨2,.cas⟩]
private def tape : Joint.Tape Unit := [.engine (.beginWrite 0 0 ())] ++
  pcs.map (fun _ => .engine (.atomic 0 0)) ++ [.engine (.beginWrite 0 1 ()),.engine (.atomic 0 1),
    .engine (.beginWrite 1 2 ()),.engine (.atomic 1 2),.issue 1 (.load a (.payload .opponent) .relaxed) 99 0]

private def calls (oldExpected : Bool) : Calls := fun k =>
  ⟨a,if k = 1 && !oldExpected then ⟨2,by decide⟩ else ⟨0,by decide⟩,.clear,k < 2,
    fun pc => (if k = 0 then 1 else if k = 1 then 8 else 10) + pc⟩

private def graph (second third payload : Nat) : Graph :=
  ⟨fun _ _ => 0,fun addr r => if addr = a then match r with
    | 1 => some tape.length | 8 => some second | 10 => some third | 11 => some payload | _ => none
    else none,fun _ _ => False⟩

-- The second writer consumes the first release. The third writer fails against
-- the second claim. The read returns the zero written to the opponent field.
example : check (graph 6 8 4) (calls false) false sites tape = true := by decide
-- The player and opponent stores have identical payloads at distinct events.
example : check (graph 6 8 3) (calls false) false sites tape = false := by decide
-- A successful CAS sees its expected resident zero, skipping two modifications.
example : check (graph tape.length 8 4) (calls true) false sites tape = false := by decide
-- A strong CAS cannot fail when its source word equals the expected operand.
example : check (graph 6 tape.length 4) (calls false) false sites tape = false := by decide

-- Site-local fixture for rank identity alone. A wrapping release and the resident
-- word both contain zero. The two recorded sites do not model the intervening
-- full writer history needed to reproduce a sequence cycle in a complete capture.
private def equalCalls : Calls := fun k =>
  ⟨a,if k = 0 then ⟨0xfffffffffffffffe,by decide⟩ else ⟨0,by decide⟩,.clear,true,fun _ => k⟩
private def equalSites : List Record := [.writer ⟨0,.finish⟩,.writer ⟨1,.cas⟩]
private def equalTape : Joint.Tape Unit := [.engine (.atomic 0 0),.engine (.atomic 1 1)]
private def equalGraph (src : Nat) : Graph := ⟨fun _ _ => 0,fun _ _ => some src,fun _ _ => False⟩

example : check (equalGraph 0) equalCalls false equalSites equalTape = true ∧
    check (equalGraph equalTape.length) equalCalls false equalSites equalTape = false := by decide

end Regression
end NeuralReversi.PvsParallel.Interleaving.GraphInstructions
