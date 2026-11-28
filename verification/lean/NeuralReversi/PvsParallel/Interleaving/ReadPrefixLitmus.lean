module
public import NeuralReversi.PvsParallel.Interleaving.GraphInstructions
public import NeuralReversi.Board.InitializationModel

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.ReadPrefixLitmus

open TranspositionTable
open SeqLock (Location)
open FrameTableReads (Address)
open NativeTableWrites.Journal (Calls Record Pc mods)
open JournalAtomicGraph (Graph Rules SB HB RF)
open ModificationIndex (decode)

/-- One lookup on worker 1 spans a partial seqlock update by worker 0. The list
    is in happens-before order; a relaxed load can read an incomparable store. -/
def addr : Address := ⟨4,0⟩
def input : PvsBoardTree.Position := ⟨Board.newBoard,rfl⟩
def calls : Calls := fun _ => ⟨addr,⟨0,rfl⟩,.store none input (Packing.unpack false 16777216),true,
  fun pc => if pc = 2 then 6 else pc+1⟩
def es : Joint.Tape Unit :=
  [.engine (.beginWrite 0 0 ()),.engine (.atomic 0 0),.engine (.atomic 0 0),
   .issue 1 (.load addr .sequence .acquire) 0 0,.issue 1 (.load addr (.payload .player) .relaxed) 0 input.val.player,
   .issue 1 (.load addr (.payload .opponent) .relaxed) 0 0,.engine (.atomic 0 0)]
def rs : List Record := [.writer ⟨0,.cas⟩,.writer ⟨0,.fence⟩,.writer ⟨0,.field .player⟩]
def graph : Graph :=
  ⟨fun _ _ => 0,fun a r => if a = addr then
    match r with | 1 | 3 => some 7 | 4 => some 6 | 5 => some 9 | _ => none
    else none,SB es⟩

theorem sites : WriterSteps.Facts calls es rs := by
  let mc : Bitboard.Moves.X86Config := ⟨false,false⟩
  let fc : Cutoffs.FlipConfig := ⟨false,false,false⟩
  let state := (CallJournal.run mc fc calls false (es.map Joint.Event.engineEvent)).get (by decide)
  exact WriterSteps.run (t := state) (mc := mc) (fc := fc) (endian := false) (by rfl)

private theorem timed : ModificationIndex.Timed calls rs es.length :=
  ⟨fun _ => sites.bound,fun _ _ => sites.injective,sites.distinct⟩

private theorem rankCases {a : Address} {w i : Nat} {l : Location}
    (found : decode calls rs es.length a w = some (l,i)) :
    (i = 0 ∧ w = 7 + ModificationIndex.tag l) ∨
    (a = addr ∧ i = 1 ∧ ((l = .sequence ∧ w = 1) ∨ (l = .payload .player ∧ w = 6))) := by
  obtain ⟨inside,stamp⟩ := (ModificationIndex.decode_eq_some_iff timed).mp found
  cases i with
  | zero => exact Or.inl ⟨rfl,stamp.symm⟩
  | succ j =>
    have hit := List.getElem?_eq_getElem (show j < _ from inside)
    obtain ⟨mem,address,_,location⟩ := NativeTableWrites.Journal.mem_mods.mp (List.mem_of_getElem? hit)
    change addr = a at address
    subst a
    have emitted := (ModificationIndex.modified (endian := false) (initial := graph.initial) (n := es.length) hit).1
    rw [emitted] at stamp
    generalize locSite : (mods calls rs addr l)[j] = site at mem location stamp
    simp only [rs,List.mem_cons,List.not_mem_nil,or_false,Record.writer.injEq] at mem
    rcases mem with cas | fence | field
    · rw [cas] at location stamp
      cases location
      have bound : j+1 ≤ 1 := inside
      exact .inr ⟨rfl,by omega,Or.inl ⟨rfl,stamp.symm⟩⟩
    · rw [fence] at location
      cases location
    · rw [field] at location stamp
      cases location
      have bound : j+1 ≤ 1 := inside
      exact .inr ⟨rfl,by omega,Or.inr ⟨rfl,stamp.symm⟩⟩

private theorem sourceCases {a : Address} {r w : Nat} (found : graph.source a r = some w) :
    (w = 7 ∧ (r = 1 ∨ r = 3)) ∨ (w = 6 ∧ r = 4) ∨ (w = 9 ∧ r = 5) := by
  dsimp only [graph] at found
  split at found <;> try contradiction
  split at found <;> cases found <;> simp

private theorem edge {u v : Nat} (h : HB graph es u v) :
    u < v ∧ v < 7 ∧ ((u ≤ 2 ∨ u = 6) ↔ (v ≤ 2 ∨ v = 6)) := by
  obtain ⟨_,_,lt,inside,same⟩ := h
  change v < 7 at inside
  refine ⟨lt,inside,?_⟩
  have group (i : Nat) (bound : i < 7) :
      Joint.worker es i = 0 ↔ i ≤ 2 ∨ i = 6 := by
    obtain h | h | h | h | h | h | h : i = 0 ∨ i = 1 ∨ i = 2 ∨ i = 3 ∨ i = 4 ∨ i = 5 ∨ i = 6 := by omega
    all_goals subst i; decide
  rw [← group u (by omega),← group v inside,same]

theorem rules : Rules graph calls false rs es where
  program := fun hb => ⟨Nat.lt_trans hb.1 hb.2.1,hb.2.1,hb⟩
  irrefl := fun _ hb => Nat.lt_irrefl _ (edge hb).1
  trans := by
    rintro u v w ⟨inside,_,lt,_,same⟩ ⟨_,outside,lt',_,same'⟩
    exact ⟨inside,outside,Nat.lt_trans lt lt',outside,same.trans same'⟩
  sync := by
    rintro a w r (⟨k,yes,_,stamp,_⟩ | ⟨k,yes,_,stamp,_⟩) _ read
    · simp [rs] at yes
    · change 1 = w at stamp
      have := sourceCases read.1
      omega
  fences := by
    intro a f w r b _ opcode
    cases b with
    | zero => contradiction
    | succ b => rcases b with _|(_|(_|(_|(_|(_|b))))) <;> contradiction
  ww := by
    intro a l u v i j x y hb
    obtain ⟨lt,inside,_⟩ := edge hb
    rcases rankCases x with ⟨_,initial⟩ | ⟨_,rfl,(⟨rfl,ev⟩ | ⟨rfl,ev⟩)⟩ <;>
      rcases rankCases y with ⟨_,initial'⟩ | ⟨_,rfl,(⟨loc,ev'⟩ | ⟨loc,ev'⟩)⟩ <;> first | contradiction | omega
  wr := by
    intro a l u w r i j x y hb read
    obtain ⟨lt,inside,group⟩ := edge hb
    have source := sourceCases read.1
    rcases rankCases x with ⟨rfl,_⟩ | ⟨_,rfl,(⟨rfl,ev⟩ | ⟨rfl,ev⟩)⟩ <;>
      rcases rankCases y with ⟨rfl,initial⟩ | ⟨_,rfl,_⟩ <;> first | decide | omega
  rr := by
    intro a l u v p r i j x y hb first last
    obtain ⟨lt,inside,group⟩ := edge hb
    have prev := sourceCases first.1
    have next := sourceCases last.1
    rcases rankCases x with ⟨rfl,_⟩ | ⟨_,rfl,(⟨rfl,ev⟩ | ⟨rfl,ev⟩)⟩ <;>
      rcases rankCases y with ⟨rfl,initial⟩ | ⟨_,rfl,_⟩ <;> first | decide | (simp [ModificationIndex.tag] at initial; omega)
  rw := by
    intro a l u v r i j x y read hb
    obtain ⟨lt,inside,group⟩ := edge hb
    have prev := sourceCases read.1
    rcases rankCases y with ⟨rfl,initial⟩ | ⟨_,rfl,(⟨rfl,ev⟩ | ⟨rfl,ev⟩)⟩ <;>
      rcases rankCases x with ⟨rfl,initial'⟩ | ⟨_,rfl,(⟨loc,ev'⟩ | ⟨loc,ev'⟩)⟩ <;> first | decide | contradiction | omega
  writer := GraphInstructions.writers_of_check (by decide +kernel)
  load r := GraphInstructions.loads_of_check (by decide +kernel) r

def mem := (JournalAtomicGraph.execution sites rules).memory addr
def ev (i : Nat) := 3+i
def source (i : Nat) := if i = 1 then 1 else 0
def lookup := ReaderMir.lookupRun false input.val.player input.val.opponent (mem.readValue ∘ ev)
def actions : List ReaderMir.AtomicAction :=
  [.load .sequence .acquire,.load (.payload .player) .relaxed,.load (.payload .opponent) .relaxed]

private theorem executed : ReaderMir.Realizes mem lookup.1 ev source := by
  change ReaderMir.Realizes mem actions ev source
  constructor
  · intro i bound
    change i+1 < 3 at bound
    have cases_ : i = 0 ∨ i = 1 := by omega
    rcases cases_ with rfl | rfl <;> exact ⟨by decide,by decide,rfl⟩
  · intro i l mode h
    rcases i with _|(_|(_|i)) <;> try contradiction
    · cases h
      exact ⟨rfl,_,_,rfl,Or.inl ⟨_,rfl⟩⟩
    · cases h
      exact ⟨rfl,Location.payload .player,1,rfl,Or.inl ⟨_,rfl⟩⟩
    · cases h
      exact ⟨rfl,Location.payload .opponent,0,rfl,Or.inl ⟨_,rfl⟩⟩
  · intro i l h
    rcases i with _|(_|(_|i)) <;> cases h
    exact Or.inl ⟨_,rfl⟩
  · intro i h
    rcases i with _|(_|(_|i)) <;> contradiction

private theorem missing (g : Graph) (resident : g.initial addr (.payload .player) = 0) :
    ¬Rules g calls false (rs.take 2) (es.take 6) := by
  intro short
  let small := es.take 6
  let mc : Bitboard.Moves.X86Config := ⟨false,false⟩
  let fc : Cutoffs.FlipConfig := ⟨false,false,false⟩
  let state := (CallJournal.run mc fc calls false (small.map Joint.Event.engineEvent)).get (by decide)
  have h : WriterSteps.Facts calls small (rs.take 2) :=
    WriterSteps.run (t := state) (mc := mc) (fc := fc) (endian := false) (by rfl)
  have stamps : ModificationIndex.Timed calls (rs.take 2) small.length :=
    ⟨fun _ => h.bound,fun _ _ => h.injective,h.distinct⟩
  obtain ⟨w,i,_,decoded,word⟩ := short.load 4 addr (.payload .player) .relaxed rfl
  have bound := ((ModificationIndex.decode_eq_some_iff stamps).mp decoded).1
  change i ≤ 0 at bound
  obtain rfl : i = 0 := by omega
  change g.initial addr (.payload .player) = input.val.player at word
  rw [resident] at word
  contradiction

/-- A lawful full writer/reader execution contains the Rust lookup's early
    key mismatch. Its first six events form an HB-downward-closed cut, but
    the player read came from the later, incomparable store at event 6. The
    truncated instruction list contains no store of the observed player word. -/
def Contract : Prop :=
  Rules graph calls false rs es ∧ ReaderMir.Realizes mem lookup.1 ev source ∧
  lookup = (actions,none) ∧ (∀ u v, HB graph es u v → v < 6 → u < 6) ∧
  mem.rf 6 4 ∧ ¬mem.hb 6 4 ∧ ¬mem.hb 4 6 ∧
  ReadCandidates.sources calls false (rs.take 2) addr (.payload .player) 0 input.val.player (fun _ => false) = [] ∧
  ∀ g : Graph, g.initial addr (.payload .player) = 0 → ¬Rules g calls false (rs.take 2) (es.take 6)

theorem witness : Contract := by
  refine ⟨rules,executed,rfl,fun _ _ hb bound => Nat.lt_trans (edge hb).1 bound,
    ⟨rfl,Location.payload .player,1,rfl,Or.inl ⟨_,rfl⟩⟩,?_,?_,rfl,missing⟩
  · intro h
    have := (edge h).1
    omega
  · intro h
    have := (edge h).2.2
    omega

end NeuralReversi.PvsParallel.Interleaving.ReadPrefixLitmus
