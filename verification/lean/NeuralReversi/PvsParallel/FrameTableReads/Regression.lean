module
public import NeuralReversi.PvsParallel.FrameTableReads.ProbeRegisters
meta import NeuralReversi.PvsParallel.FrameTableReads.Execution
meta import NeuralReversi.PvsParallel.FrameTableReads.ProbeRegisters
meta import NeuralReversi.PvsParallel.FrameTableReads.Schedule

@[expose] public section

namespace NeuralReversi.PvsParallel.FrameTableReads.Operations.Regression

open TranspositionTable
open Execution (Call Caller Turn)

private def board : PvsBoardTree.Position := ⟨⟨33024,66⟩,rfl⟩
private def word (xs : List Nat) (i : Nat) := BitVec.ofNat 64 (xs[i]?.getD 0)
private def call (method : Execution.Method) (key : Nat) (values : List Nat) : Call :=
  ⟨method,0,16,BitVec.ofNat 64 key,false,⟨fun _ => word values,fun _ i => (word values i).setWidth 8⟩,id,id⟩

private def label : Execution.Instruction → String
  | .generation _ => "G"
  | .spin => "S"
  | .fence .acquire => "F"
  | .load a .sequence .acquire => s!"{a.index}A"
  | .load a .sequence .relaxed => s!"{a.index}Q"
  | .load a (.payload .player) .relaxed => s!"{a.index}P"
  | .load a (.payload .opponent) .relaxed => s!"{a.index}O"
  | .load a (.payload .metadata) .relaxed => s!"{a.index}D"
  | _ => "unexpected"

private def probeCase (key : Nat) (values : List Nat) :=
  let c := call (.probe (Cutoffs.key board)) key values
  let out := probe c.supply c.allocation c.clusters c.key c.endian (Cutoffs.key board) 0
  let slot := match out.result with | .hit s _ | .miss s => s.val
  (String.intercalate " " (c.records.map (label ∘ Execution.Atomic.instruction)),slot,(c.returned.map (fun d => (Packing.pack false d).toNat)))

private def lookupCase (key : Nat) (values : List Nat) :=
  let c := call (.lookup board) key values
  (String.intercalate " " (c.records.map (label ∘ Execution.Atomic.instruction)),(c.returned.map (fun d => (Packing.pack false d).toNat)))

/- Both slots stayed busy for all three tries. -/
#guard probeCase 0
  [1,1,0,1,1,0,1,1,0] ==
  ("0A 1A S 0A 1A S 0A 1A S",0,none)

/- The first slot became stable after one spin. -/
#guard probeCase 12682156566673660604
  [1,1,0,2,107523458334588905,33024,66,0,2] ==
  ("22A 23A S 22A 22D 22P 22O F 22Q",0,some 107523458334588905)

private def replayHint (values : List Nat) (edit : List Execution.Atomic → List Execution.Atomic := id) :=
  let q := Cutoffs.key board
  let c := call (.probe q) 0 values
  (Execution.readProbe c.allocation c.clusters c.key c.endian q (edit c.records)).map ReaderMir.bestMove

/- A matching snapshot may name an occupied square (8); a missing fence invalidates the replay. -/
private def matching (sq : Nat) := [2,2^24 + sq*2^16,33024,66,0,2]
#guard replayHint (matching 8) == some (some 8)
#guard replayHint (matching 63) == some (some 63)
#guard replayHint (matching 64) == some none
#guard replayHint (matching 8) (fun rows => rows.eraseIdx 4) == none
#guard replayHint [1,1,0,1,1,0,1,1,0] == some none

/- Only the second victim was stable at exhaustion. -/
#guard probeCase 7205759403792793600
  [1,18446744073709551614,72339167850201072,33025,64,0,0,0,1,0,72339167850201072,33025,64,0,0,0,1,0,72339167850201072,33025,64,0,0,0] ==
  ("12A 13A 13D 13P 13O F 13Q S 12A 13A 13D 13P 13O F 13Q S 12A 13A 13D 13P 13O F 13Q S",1,none)

/- A changed validation word forced another round before the victim comparison. -/
#guard probeCase 12682156566673660604
  [4,107523458334588905,33025,66,0,4,4,72339167850201072,33025,66,0,6,0,4,107523458334588905,33025,66,0,4,6,72339167850201072,33025,66,0,6,0] ==
  ("22A 22D 22P 22O F 22Q 23A 23D 23P 23O F 23Q S 22A 22D 22P 22O F 22Q 23A 23D 23P 23O F 23Q G",1,none)

/- Equal replacement scores select the first slot. -/
#guard probeCase 3530822107858468864
  [4,72057628448915433,33024,64,0,4,4,72057628448980976,33024,64,0,4,127] ==
  ("6A 6D 6P 6O F 6Q 7A 7D 7P 7O F 7Q G",0,none)

/- The second slot wrapped during lookup validation. -/
#guard lookupCase 7205759403792793600
  [1,18446744073709551614,33024,66,72339167850201072,0,0] ==
  ("12A 13A 13P 13O 13D F 13Q",none)

private def secondSlot := call (.lookup board) 7205759403792793600
  [1,18446744073709551614,33024,66,72339167850201072,0,0]
#guard secondSlot.shortSlotHistories (fun a => if a.index == 13 then SeqLock.modulus - 1 else SeqLock.modulus)
#guard !(secondSlot.shortSlotHistories (fun a => if a.index == 13 then SeqLock.modulus else 0))
#guard (call (.lookup board) 0 [1,1]).shortSlotHistories (fun _ => SeqLock.modulus)

/- A key mismatch in the first slot was followed by a matching second slot. -/
#guard lookupCase 0
  [4,33025,66,4,33024,66,72339167850201072,0,4] ==
  ("0A 0P 0O 1A 1P 1O 1D F 1Q",some 72339167850201072)

/- Two callers with the same opening interleave with a writer. The first
    completes a snapshot while the second has only issued its initial load. -/
private def left : Caller := ⟨0,17⟩
private def right : Caller := ⟨1,17⟩
private def turns : List Turn := [.reader left,.reader right,.reader left,.search (.writer ⟨2,.finish⟩),.reader left,.reader left]
private def pool := fun _ : Caller => call (.probe (Cutoffs.key board)) 0 [0,0,0,0]
private def mixed := Execution.dispatch pool (fun _ => 0) turns

#guard (Execution.selected left mixed).map (label ∘ Execution.Atomic.instruction) == ["0A","0D","F","0Q"]
#guard (Execution.selected right mixed).map (label ∘ Execution.Atomic.instruction) == ["0A"]
#guard (Execution.journal mixed).length == 1

end NeuralReversi.PvsParallel.FrameTableReads.Operations.Regression
