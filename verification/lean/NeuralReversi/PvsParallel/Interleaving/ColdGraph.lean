module
public import NeuralReversi.PvsParallel.Interleaving.JournalAtomicGraph

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.ColdGraph

open TranspositionTable
open SeqLock (Location)
open FrameTableReads (Address)
open NativeTableWrites.Journal (Calls Record Site mods)
open JournalAtomicGraph (Graph Rules SB)

private theorem absent (empty : ∀ s, Record.writer s ∉ rs) (c : Calls) (a : Address) (l : Location) :
    mods c rs a l = [] := by
  apply List.eq_nil_iff_forall_not_mem.mpr
  exact fun _ mem => empty _ (NativeTableWrites.Journal.mem_mods.mp mem).1

private theorem timed (empty : ∀ s, Record.writer s ∉ rs) : ModificationIndex.Timed c rs n :=
  ⟨fun s present => (empty s present).elim,fun s _ present => (empty s present).elim,
   fun a l => by simp [absent empty c a l]⟩

/-- For a freshly zeroed table without an intervening writer, the only physical source
    for a load is the addressed location's initialization. -/
def graph (es : Joint.Tape Op) : Graph where
  initial := fun _ _ => 0
  source a r := do
    let some (.load address loc _) := Joint.opcode es r | none
    guard (address = a)
    return es.length + ModificationIndex.tag loc
  hb := SB es

private theorem initial (empty : ∀ s, Record.writer s ∉ rs)
    (found : ModificationIndex.decode c rs n a w = some (l,i)) : i = 0 ∧ w = n + ModificationIndex.tag l := by
  obtain ⟨valid,atWord⟩ := (ModificationIndex.decode_eq_some_iff (timed empty)).mp found
  rw [absent empty] at valid
  obtain rfl : i = 0 := Nat.eq_zero_of_le_zero valid
  exact ⟨rfl,atWord.symm⟩

theorem rules (c : Calls) (endian : Bool) (rs : List Record) (es : Joint.Tape Op)
    (empty : ∀ s, Record.writer s ∉ rs)
    (zero : ∀ r a l mode, Joint.opcode es r = some (.load a l mode) → Joint.word es r = 0) :
    Rules (graph es) c endian rs es where
  trans := by
    rintro u v w ⟨_,_,uv⟩ ⟨_,_,vw⟩
    exact ⟨Nat.lt_trans uv.1 (Nat.lt_trans vw.1 vw.2.1),vw.2.1,
      Nat.lt_trans uv.1 vw.1,vw.2.1,uv.2.2.trans vw.2.2⟩
  irrefl := fun _ h => Nat.lt_irrefl _ h.2.2.1
  program := fun h => ⟨Nat.lt_trans h.1 h.2.1,h.2.1,h⟩
  sync := by rintro a w r (⟨k,present,_⟩|⟨k,present,_⟩) <;> exact (empty _ present).elim
  fences := by rintro a f w r b ⟨k,present,_⟩; exact (empty _ present).elim
  ww := by
    intro a l u v i j first _ edge
    have source := (initial empty first).2
    have early := edge.1
    omega
  wr := by intro a l u w r i j first _ _ _; rw [(initial empty first).1]; exact Nat.zero_le _
  rr := by intro a l u v p r i j first _ _ _ _; rw [(initial empty first).1]; exact Nat.zero_le _
  rw := by
    intro a l u v r i j _ final _ edge
    have source := (initial empty final).2
    have early := edge.2.1
    omega
  writer := fun s present => (empty s present).elim
  load := by
    intro r a l mode issued
    exact ⟨es.length + ModificationIndex.tag l,0,by simp [graph,issued,guard],
      ModificationIndex.decode_initial (timed empty) a l,(zero _ _ _ _ issued).symm⟩

end NeuralReversi.PvsParallel.Interleaving.ColdGraph
