import Lean.Data.Json
import NeuralReversi.PvsParallel.Interleaving.AtomicReadback
open Lean NeuralReversi NeuralReversi.PvsParallel
open TranspositionTable NativeTableWrites.Journal Interleaving

structure Observation where
  workers : List Nat
  words : List Nat
  cut : Nat
  source : Nat
  candidates : List Nat
  deriving FromJson

private def addr : FrameTableReads.Address := ⟨1,0⟩
private def loc : SeqLock.Location := .payload .player
private def pcs : List Pc := [.cas,.fence,.field .player,.field .opponent,.field .metadata,.finish]

private def attempt (o : Observation) (j : Nat) : Attempt :=
  let player := BitVec.ofNat 64 (o.words[j+1]?.getD 0)
  let b : PvsBoardTree.Position := ⟨⟨player,0⟩,by simp [Board.wellFormed]⟩
  let seq := (BitVec.ofNat 64 j) <<< 1
  have seqEven : seq &&& 1 = 0 := by
    apply BitVec.eq_of_toNat_eq
    simp [seq, Nat.and_one_is_mod, Nat.shiftLeft_eq]
  ⟨addr,⟨seq,seqEven⟩,.store none b (Packing.unpack false 0),true,
    fun pc => 7*j+1+pc+(if j < o.cut then 0 else 1)⟩

private def writes (o : Observation) (j : Nat) : List (Joint.Event Unit) :=
  let w := o.workers[j]?.getD 0
  .engine (.beginWrite w j ()) :: pcs.map (fun _ => .engine (.atomic w j))

private def replay (o : Observation) (mode : String) : Except String (List Nat) := do
  unless o.workers.length == 3 && o.words.length == 4 && o.words.head? == some 0 &&
      o.cut ≤ 3 && o.source ≤ 3 && (o.workers ++ o.words).all (· ≤ 1) do
    throw "invalid one-word transcript"
  let ids := List.range o.workers.length
  let records := ids.flatMap fun j => pcs.map fun pc => Record.writer ⟨j,pc⟩
  let seen := BitVec.ofNat 64 (o.words[o.source]?.getD 0)
  -- Deliberately impossible source label. Reconstruction reads only the word and tape.
  let load : Joint.Event Unit := .issue 0 (.load addr loc .relaxed) 99 seen
  let tape := (ids.take o.cut).flatMap (writes o) ++ [load] ++ (ids.drop o.cut).flatMap (writes o)
  let r := 7 * o.cut
  let exact := AtomicReadback.candidates (attempt o) false records tape r addr loc 0
  unless exact.contains o.source do throw "Lean excluded the semantic source"
  let sanity := AtomicReadback.uniqueSource (attempt o) false records tape (fun _ _ => 0) r
  unless sanity == (match exact with | [i] => some i | _ => none) do
    throw "Lean singleton decoding disagrees"
  let decoded ← match mode with
    | "step" => pure exact
    | "latestOnly" => pure (exact.filter (· == o.cut))
    | "omitUpper" => pure (ReadCandidates.onTape (attempt o) false records tape r addr loc 0)
    | _ => throw "unknown replay policy"
  unless o.candidates == decoded do throw "candidate indices disagree with Lean"
  unless (exact == o.candidates) == (mode == "step") do throw "expected a decoder difference"
  return exact

def main (args : List String) : IO Unit := do
  let [mode,path] := args | throw (IO.userError "expected policy and final TLC state")
  let raw ← IO.FS.readFile path
  let outcome : Except String (List Nat) := do
    let o : Observation ← fromJson? (← Json.parse raw)
    replay o mode
  match outcome with
  | .ok indices => IO.println s!"Lean readers retained {indices}"
  | .error reason => throw (IO.userError reason)
