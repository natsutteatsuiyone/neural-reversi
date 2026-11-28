module
public import NeuralReversi.EndgameSearch.DeepRecursionMir
public import NeuralReversi.EndgameSearch.CacheMir

@[expose] public section

namespace NeuralReversi.EndgameSearch.RecursionCacheMir

open EndgameTreeCache (Address Store Table entryFor soundHistory)
open CacheState (Cells represents)
open Flip.Dispatch (TargetConfig)
open Bitboard.Moves (X86Config)
open PvsBoardTree (Position)
open Cancellation (EmptyState State)

def Reads (width : Nat) (hash : Board.Board → BitVec 64) (address : Address)
    (past : List Store) (cells : Cells) : Prop :=
  ∀ t b a, CacheMir.probe (cells t) (CacheMir.slot width (cells t).size (hash b)) b a =
    some (EndgameCache.probe (entryFor address past t b) b a)

inductive Steps (width : Nat) (hash : Board.Board → BitVec 64) (address : Address) :
    List Store → Cells → List Store → Cells → Prop
  | done (reads : Reads width hash address past cells) : Steps width hash address past cells [] cells
  | more (reads : Reads width hash address past cells) (writes : CacheMir.writeCells width hash cells w = some next)
      (tail : Steps width hash address (past ++ [w]) next ws final) : Steps width hash address past cells (w :: ws) final

private theorem replay (width : Nat) (length : Table → Nat) (hash : Board.Board → BitVec 64)
    (fit : CacheMir.Fits width length) (past : List Store) (cells : Cells)
    (shape : ∀ t, (cells t).size = length t) (prior : represents (CacheMir.address width length hash) past cells)
    (writes : List Store) :
    ∃ final, Steps width hash (CacheMir.address width length hash) past cells writes final := by
  have reads : Reads width hash (CacheMir.address width length hash) past cells := by
    intro t b a
    have atSlot := prior t b
    simp only [CacheMir.address] at atSlot
    simp only [CacheMir.probe, shape, atSlot, Option.map_some, CacheMir.probeCell_eq]
  cases writes with
  | nil => exact ⟨cells,.done reads⟩
  | cons w rest =>
    have written := CacheMir.write_eq (hash := hash) w shape fit
    obtain ⟨next, newShape⟩ := CacheMir.written_represents w shape fit prior written
    obtain ⟨final, steps⟩ := replay width length hash fit _ _ newShape next rest
    exact ⟨final,.more reads written steps⟩

theorem shallow (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : FlipConfig)
    (width : Nat) (length : Table → Nat) (hash : Board.Board → BitVec 64) (fit : CacheMir.Fits width length)
    (b : Position) (s : EmptyState) (a : Int) (past : List Store) (cells : Cells)
    (aligned : s.matches b) (safe : SafeThreshold a) (valid : soundHistory mc fc past)
    (shape : ∀ t, (cells t).size = length t) (prior : represents (CacheMir.address width length hash) past cells) :
    let addr := CacheMir.address width length hash
    let out := ShallowMir.run target bmi2 mc fc addr b s a past
    ∃ ws final, past ++ ws = out.history ∧ Steps width hash addr past cells ws final := by
  let addr := CacheMir.address width length hash
  have eq := ShallowMir.agrees target bmi2 mc fc addr b s a past aligned safe valid
  obtain ⟨_, _, _, ws, after⟩ := contract_proven mc fc (orderedOptions target bmi2 mc fc addr) .shallow b a past
    (orderedOptions_correct target bmi2 mc fc addr) valid (fun _ _ => safe)
  obtain ⟨final, steps⟩ := replay width length hash fit past cells shape prior ws
  exact ⟨ws, final, after.trans (congrArg Result.history eq.symm), steps⟩

theorem deep (target : TargetConfig) (bmi2 paired : Bool) (mc : X86Config) (fc : FlipConfig)
    (width : Nat) (length : Table → Nat) (hash : Board.Board → BitVec 64) (fit : CacheMir.Fits width length)
    (b : Position) (s : State) (a : Int) (cells : Cells)
    (aligned : s.empties.matches b) (safe : SafeThreshold a) (valid : soundHistory mc fc s.history)
    (shape : ∀ t, (cells t).size = length t)
    (prior : represents (CacheMir.address width length hash) s.history cells) :
    let addr := CacheMir.address width length hash
    let out := DeepRecursionMir.run target bmi2 paired mc fc addr b a s
    ∃ ws final, s.history ++ ws = out.state.history ∧ Steps width hash addr s.history cells ws final := by
  obtain ⟨_, _, ⟨ws, after⟩, _⟩ := DeepRecursionMir.returned_score target bmi2 paired mc fc
    (CacheMir.address width length hash) b a s aligned safe valid
  obtain ⟨final, steps⟩ := replay width length hash fit s.history cells shape prior ws
  exact ⟨ws, final, after, steps⟩

end NeuralReversi.EndgameSearch.RecursionCacheMir
