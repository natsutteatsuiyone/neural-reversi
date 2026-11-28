module
public import NeuralReversi.PvsParallel.SerialTable.Reads
public import NeuralReversi.PvsParallel.SerialTable.Store
public import NeuralReversi.PvsParallel.SerialTable.Writes
public import NeuralReversi.PvsParallel.SerialCacheStack
public import NeuralReversi.EndgameSearch.SerialLeafCache

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialTable

open EndgameSearch.CacheHistory (Pool Local)
open EndgameSearch.SerialLeafCache (Reachable)
open SerialExecution (Inputs)
open PvsBoardTree (Position minimax)
open TranspositionTable (Depth)

structure Memory (l : Layout) where
  tt : Table l
  supported : tt.image.Supported tt.history
  cache : Local

def Memory.Good (p : Pool) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (m : Memory l) : Prop :=
  m.tt.Good mc fc ∧ Reachable p mc fc m.cache

def Memory.empty (l : Layout) (g : BitVec 8) (p : Pool) : Memory l :=
  ⟨Table.empty l g, Table.empty_supported l g, p.empty⟩

theorem Memory.empty_good (l : Layout) (g : BitVec 8) (p : Pool) (mc fc) :
    (Memory.empty l g p).Good p mc fc :=
  ⟨Table.empty_good l g mc fc,EndgameSearch.SerialLeafCache.cold p mc fc⟩

def Memory.inputs (m : Memory l) (c : SerialScout.Config) (base : Inputs mc fc) : Inputs mc fc :=
  SerialCacheStack.source (m.tt.image.inputs m.supported c base) (fun _ => m.cache.history)

theorem Memory.facts (m : Memory l) (c : SerialScout.Config) (base : Inputs mc fc) (good : m.Good p mc fc) :
    SerialExecution.ReadFacts (m.inputs c base) ∧ SerialTermination.Hints (m.inputs c base) := by
  have shared (limit : Nat) := m.tt.image.inputs_shared (limit := limit) m.supported c base
    good.1.publications good.1.hints
  refine ⟨⟨fun _ _ _ _ _ => good.2.1,?_,?_⟩,?_⟩
  · intro i ctx role depth
    exact (shared ctx.request.depth.toNat).probe i ctx role depth (Nat.le_refl _)
  · intro i ctx role depth
    exact (shared ctx.request.depth.toNat).table i ctx role depth (Nat.le_refl _)
  · intro i ctx role depth
    exact (shared ctx.request.depth.toNat).hints i ctx role depth (Nat.le_refl _)

def Memory.apply (m : Memory l) (slot : l.Index) (proposed : Option Writes.Proposal)
    (level : TranspositionTable.Selectivity := Cutoffs.exactSelectivity) : Memory l :=
  match proposed with
  | none => m
  | some w =>
    let t := m.tt.store slot w.board (Writes.word w.score) w.bound w.depth w.move level true w.ttPv
    ⟨t,m.tt.store_supported m.supported _ _ _ _ _ _ _ _ _,m.cache⟩

theorem Memory.apply_good (m : Memory l) (good : m.Good p mc fc) (slot : l.Index)
    {b : Position} {d : Depth} {pv : Bool} {a z v : Int} {proposed : Option Writes.Proposal}
    (shape : ∀ w, proposed = some w → ∃ tag, Writes.Produced b d pv tag a z v w)
    (window : Aspiration.Window a z) (sound : Aspiration.SoundResult (minimax mc fc b) a z v) :
    (m.apply slot proposed).Good p mc fc := by
  cases proposed with
  | none => exact good
  | some w =>
    obtain ⟨_,h⟩ := shape w rfl
    have repr (v : Int) (lower : -32767 ≤ v) (upper : v ≤ 32767) : (Writes.word v).raw.toInt = v :=
      BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
    obtain ⟨lo,ordered,hi⟩ := window
    have av := repr _ lo (by omega)
    have zv := repr _ (by omega) hi
    have vv := repr v (by have := sound.lower; omega) (by have := sound.upper; omega)
    refine ⟨?_,good.2⟩
    change (m.tt.store slot w.board (Writes.word w.score) w.bound w.depth w.move _ true w.ttPv).Good mc fc
    rw [h.board,h.value,h.depth,h.bound]
    exact m.tt.store_searched good.1 slot b pv w.ttPv _ _ _ d w.move
      (by rw [av,zv]; exact ⟨lo,ordered,hi⟩) (by rw [av,zv,vv]; exact sound) h.hint

end NeuralReversi.PvsParallel.SerialTable
