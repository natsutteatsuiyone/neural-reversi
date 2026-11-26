import NeuralReversi.TranspositionTable.ReaderMir.ProbeProgram
import NeuralReversi.TranspositionTable.ReaderMir.Lookup

namespace NeuralReversi.TranspositionTable.ReaderMir

open ProbeRetry (Observation Attempt RawAttempt Result)

def Pass.result (p : Pass) : Option Result := match p.exit with
  | .done r => some r | .retry _ => none

theorem pass_decisive (a : Attempt) :
    (probePass a.generation a.first a.second).result = ProbeRetry.decisive a := by
  obtain ⟨first,second,g⟩ := a
  cases first <;> cases second <;> rfl

theorem pass_scan (a : Attempt) (rest : List Attempt) (fallback : Fin 2) :
    ProbeRetry.scan fallback (a :: rest) =
      match (probePass a.generation a.first a.second).exit with
      | .done v => v | .retry stable => ProbeRetry.scan (stable.getD fallback) rest := by
  obtain ⟨first,second,g⟩ := a
  cases first <;> cases second <;> rfl

def readsSecond : Observation → Bool | .busy | .other _ => true | _ => false

def readsGeneration : Observation → Observation → Bool
  | .other _,.other _ => true | _,_ => false

theorem second_of_snapshot (query : ProbeRetry.Key) (snap : Option (ProbeRetry.Key × EntryData)) :
    readsSecond (ProbeRetry.classify query snap) = probeSecond query snap := by
  cases snap with
  | none => rfl
  | some pair =>
    obtain ⟨stored,data⟩ := pair
    by_cases empty : data.bound = 0#8 <;> cases eq : sameBoardBits stored.1 stored.2 query.1 query.2 <;>
      simp [readsSecond,ProbeRetry.classify,probeSecond,empty,eq]

theorem pass_effects (a : Attempt) :
    (probePass a.generation a.first a.second).effects =
      [.read 0] ++ (if readsSecond a.first then [.read 1] else []) ++
      (if readsGeneration a.first a.second then [.generation] else []) ++
      (if (ProbeRetry.decisive a).isNone then [.spin] else []) := by
  obtain ⟨first,second,g⟩ := a
  cases first <;> cases second <;> rfl

/-- A fresh generation read happens only when both slots are occupied misses.
    On every other path its input is immaterial. -/
theorem generation_unused (a b : Observation) (noLoad : readsGeneration a b = false) (g g' : BitVec 8) :
    probePass g a b = probePass g' a b := by
  cases a <;> cases b <;> first | rfl | contradiction

end NeuralReversi.TranspositionTable.ReaderMir
