import NeuralReversi.PvsParallel.Interleaving.SeedTable
import NeuralReversi.PvsParallel.Interleaving.Interior
import NeuralReversi.PvsParallel.Interleaving.RootProgram

namespace NeuralReversi.PvsParallel.Interleaving.SeedHistory

open CausalTable (Values Lookups Frame)
open PvsBoardTree (Position play)
open NativeTableWrites.Journal (Even Record)
open TranspositionTable
open AspirationMir
open AspirationMir.SerialTable.PvInterior (InsideRange)
open SerialCalls.ScoreWords (Finite)
open Aspiration (Window)

/-- Only the unrestricted root line is eligible to publish a whole-board TT bound.
    Its registers, stage and result come from one invocation of the root program. -/
inductive Body (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (level : Selectivity)
    (prior : Values) (reads : Lookups) (b : Position) : Bool → Int → Int → Int → Prop where
  | internal (frame : WeakFrame.Frame mc fc prior reads b pv a z v) : Body mc fc level prior reads b pv a z v
  | boundary (p : EndgameSearch.CacheHistory.Pool) (timeline : EndgameSearch.CacheHistory.Timeline p)
      (worker opening : Nat) (i : EndgameSearch.CacheHistory.Input) (board : i.board = b)
      (win : Window i.alpha.raw.toInt i.beta.raw.toInt) (narrow : i.beta.raw.toInt = i.alpha.raw.toInt+1)
      (arm : i.arm = .deep i.active) (score : (i.run mc fc p.address (timeline.readAt mc fc worker opening).history).score = some v) :
      Body mc fc level prior reads b false i.alpha.raw.toInt i.beta.raw.toInt v
  | root (s : SerialStack.State)
      (calls : Selectivity → Option (RootProgram.Operands mc b (BoardFrame.Children mc fc b prior) s.regs.beta.toInt))
      (cover : Root.Covers mc b (s.rows.map RootMoveStorage.Row.square)) (inside : 0 < s.rows.length)
      (window : Window s.regs.alpha.toInt s.regs.beta.toInt)
      (run : (calls level).bind (RootProgram.run fc 0 s) = some ans) :
      Body mc fc level prior reads b true s.regs.alpha.toInt s.regs.beta.toInt ans.value

def Word (pv : Bool) (a z v : Int) : Prop := Finite v ∧ (Window a z → pv = true → InsideRange a z v)

private theorem Body.word (run : Body mc fc level prior reads b pv a z v)
    (earlier : ∀ b p a z v, prior b p a z v → Word p a z v) (table : ∀ b e, reads b e → Finite e.score.toInt) :
    Word pv a z v := by
  have finite := fun b p a z v h => (earlier b p a z v h).1
  have interior := fun b p a z v win h => (earlier b p a z v h).2 win
  cases run with
  | internal node =>
    exact ⟨WeakFrame.preserves _ _ finite table _ _ _ _ _ node,fun w yes => Interior.weak node w yes finite interior⟩
  | boundary p timeline worker opening i board win narrow arm result =>
    unfold EndgameSearch.CacheHistory.Input.run at result
    rw [arm] at result
    have bound := DeepLeafMir.local_bound i.target i.bmi2 i.paired mc fc p.address i.board
      i.alpha i.beta (i.state _) i.active i.checked i.aborted i.aligned (timeline.reads mc fc worker opening).1 narrow result
    have sound := bound_sound (PvsBoardTree.build mc fc i.board) win bound
    exact ⟨⟨sound.lower,sound.upper⟩,fun _ yes => Bool.noConfusion yes⟩
  | root s calls cover pos win found =>
    obtain ⟨op,_,hit⟩ := Option.bind_eq_some_iff.mp found
    cases RootProgram.ran op hit win with
    | searched trial =>
      exact ⟨(trial.finite cover pos (fun _ => finite _)).1,fun win _ =>
        Interior.root trial win (fun _ => finite _) (fun _ p a z v w used => interior _ p a z v w used)⟩
    | wiped =>
      change Word true _ _ PvsMoveLoop.scoreMax
      exact ⟨by constructor <;> decide,fun _ _ _ _ => by constructor <;> decide⟩

/-- A classifier-produced store identifies the chosen score at its own level.
    The forced lower store of ETC/presearch cannot supply an exact-bound root seed. -/
inductive StoreInput (picked : Selectivity → Values) (b : Position) : EntryData → Prop where
  | classified (slot : Nat) (score lo hi : Score) (pv : Bool) (d : Depth) (move : Option Bitboard.Square)
      (level : Selectivity) (gen : BitVec 8) (phase tag : Bool) (win : Window lo.raw.toInt hi.raw.toInt)
      (source : picked level b pv lo.raw.toInt hi.raw.toInt score.raw.toInt)
      (entered : WriterMir.storeInput slot b score (classify pv score lo hi) d move level gen phase tag = some (slot,b,e)) :
      StoreInput picked b e
  | midgame (slot : Nat) (score : Score) (d : Depth) (move : Option Bitboard.Square)
      (level : Selectivity) (gen : BitVec 8) (tag : Bool) (bounded : SeedTable.Range score.raw.toInt)
      (entered : WriterMir.storeInput slot b score .exact d move level gen false tag = some (slot,b,e)) : StoreInput picked b e
  | other (slot : Nat) (score : Score) (bound : Bound) (d : Depth) (move : Option Bitboard.Square)
      (level : Selectivity) (gen : BitVec 8) (phase tag : Bool) (oneSided : bound ≠ .exact)
      (entered : WriterMir.storeInput slot b score bound d move level gen phase tag = some (slot,b,e)) : StoreInput picked b e

theorem StoreInput.passed (input : StoreInput picked b e) : FiniteTable.Passed b e := by
  cases input with
  | classified slot score lo hi pv d m level g phase tag _ _ h => exact ⟨_,_,_,_,_,_,_,_,_,h⟩
  | midgame slot score d m level g tag _ h => exact ⟨_,_,_,_,_,_,_,_,_,h⟩
  | other slot score bound d m level g phase tag _ h => exact ⟨_,_,_,_,_,_,_,_,_,h⟩

private theorem StoreInput.range (input : StoreInput picked b e)
    (known : ∀ level b pv a z v, picked level b pv a z v → Word pv a z v) : SeedTable.Incoming b e := by
  cases input with
  | classified slot score lo hi pv d m level g phase tag win source entered =>
    apply SeedTable.ofStore _ entered
    intro full
    obtain ⟨yes,low,high⟩ := ((classify_contract ..).2.1).mp full
    exact (known _ _ _ _ _ _ source).2 win yes low high
  | midgame slot score d m level g tag bounded entered => exact SeedTable.ofStore (fun _ => bounded) entered
  | other slot score bound d m level g phase tag side entered => exact SeedTable.ofStore (fun h => (side h).elim) entered

/-- Levels decorate the scored native frame; lookup, decision and return all retain
    the event coordinates used elsewhere by the same atomic execution. -/
structure Prefix (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (x : NativeTableWrites.Journal.Execution c endian rs) (init : ∀ a, Even (x.memory a))
    (frames : List Frame) (level : Frame → Selectivity) (before : Nat → Nat → Prop) : Prop where
  irrefl : ∀ t, ¬before t t
  trans : ∀ {i j k}, before i j → before j k → before i k
  left : ∀ f ∈ frames, ∀ t, f.returned = some t → before f.chosen t
  body : ∀ f ∈ frames, Body mc fc (level f)
    (CausalTable.returned before (frames.filter (fun g => level g == level f)) f.chosen) (CausalTable.lookup x init f.chosen)
    f.board f.pv f.alpha f.beta f.score
  writes : ∀ k, Record.writer ⟨k,.finish⟩ ∈ rs → (c k).success = true →
    Table.Operands endian
      (StoreInput (fun lv => CausalTable.chosen before (frames.filter (fun g => level g == lv)) ((c k).event 5)))
      (Table.Seen x (c k).address (init _) ((c k).event 5)) (c k).origin

variable {x : NativeTableWrites.Journal.Execution c endian rs} {init : ∀ a, Even (x.memory a)}

theorem Prefix.atLevel (history : Prefix mc fc x init frames level before) (lv : Selectivity) :
    FiniteTable.Prefix x init (frames.filter (fun f => level f == lv)) before (Body mc fc lv) := by
  refine ⟨history.irrefl,history.trans,fun f member => history.left f (List.mem_filter.mp member).1,?_⟩
  intro f member
  obtain ⟨listed,same⟩ := List.mem_filter.mp member
  have tag : level f = lv := beq_iff_eq.mp same
  simpa only [tag] using history.body f listed

theorem Prefix.finiteStores (history : Prefix mc fc x init frames level before) : FiniteTable.Stores x init :=
  ⟨fun k written ok => (history.writes k written ok).mapInputs (fun _ _ input => input.passed)⟩

theorem scores (history : Prefix mc fc x init frames level before)
    (seed : ∀ a, FiniteTable.Words endian ((x.slot a (init a)).payloadAt 0)) :
    ∀ f ∈ frames, Word f.pv f.alpha f.beta f.score := by
  let atEvent (t : Nat) := ∀ f ∈ frames, f.chosen = t → Word f.pv f.alpha f.beta f.score
  have proved : ∀ t ∈ frames.map Frame.chosen, atEvent t := by
    apply CausalTable.finiteInduction (frames.map Frame.chosen) before atEvent history.irrefl history.trans
    intro t _ ih f member same
    subst t
    apply (history.body f member).word _ (fun _ _ hit => FiniteTable.lookup history.finiteStores seed hit)
    intro b p a z v used
    obtain ⟨g,inside,ret,returned,earlier,rfl,rfl,rfl,rfl,rfl⟩ := used
    have mem := (List.mem_filter.mp inside).1
    exact ih _ (List.mem_map.mpr ⟨g,mem,rfl⟩) (history.trans (history.left g mem ret returned) earlier) g mem rfl
  exact fun f mem => proved _ (List.mem_map.mpr ⟨f,mem,rfl⟩) f mem rfl

theorem stores (history : Prefix mc fc x init frames level before)
    (seed : ∀ a, FiniteTable.Words endian ((x.slot a (init a)).payloadAt 0)) : SeedTable.Stores x init := by
  have valid := scores history seed
  constructor
  intro k issued success
  apply (history.writes k issued success).mapInputs
  intro b e input
  apply input.range
  intro lv board pv a z v used
  obtain ⟨f,inside,_,rfl,rfl,rfl,rfl,rfl⟩ := used
  exact valid f (List.mem_filter.mp inside).1

/-- A reused table can center the next public aspiration search after any of the
    modeled earlier levels, even when the chosen score was published by another worker. -/
theorem estimate_bounded (history : Prefix mc fc x init frames level before)
    (finite : ∀ a, FiniteTable.Words endian ((x.slot a (init a)).payloadAt 0))
    (seed : ∀ a, SeedTable.Words endian ((x.slot a (init a)).payloadAt 0))
    (r : CausalReads.Reader) (consumer allocation : Nat) (clusters key : SeqLock.Word) (start : Nat)
    (raw : Position → Int) (b : Position) :
    let out := FrameTableReads.Operations.probe (r.supply x) allocation clusters key endian (Cutoffs.key b) start
    (∀ round ∈ out.rounds, CausalReads.Issued x r consumer round.front (FrameTableReads.snapCode endian) ∧
      ∀ back ∈ round.back, CausalReads.Issued x r consumer back (FrameTableReads.snapCode endian)) →
    SeedTable.Range (SeedTable.estimate mc fc raw b out.result).toInt :=
  SeedTable.estimate_bounded (stores history finite) seed r consumer allocation clusters key start mc fc raw b

end NeuralReversi.PvsParallel.Interleaving.SeedHistory
