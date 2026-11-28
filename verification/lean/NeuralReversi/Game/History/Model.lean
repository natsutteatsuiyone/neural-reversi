module
public import NeuralReversi.Game.History.VecProperties

@[expose] public section

namespace NeuralReversi.Game.History.VecView

structure StoredState (entryBytes : Nat) where
  board : Board
  side : Disc
  history : CheckedVec entryBytes

def projectState {entryBytes : Nat} (state : StoredState entryBytes) : ValueState :=
  project state.board state.side state.history.val

def finish {entryBytes : Nat} (entry : Entry) (history : CheckedVec entryBytes) : StoredState entryBytes :=
  ⟨entry.board,entry.side,history⟩

def undo (entryBytes : Nat) (state : StoredState entryBytes) : Bool × StoredState entryBytes :=
  let first := typedPop entryBytes state.history
  match first.1 with
  | none => (false,{state with history := first.2})
  | some entry =>
    match entry.action with
    | .automaticPass =>
      let second := typedPop entryBytes first.2
      (true,finish (second.1.getD entry) second.2)
    | .play _ | .explicitPass => (true,finish entry first.2)

def confirmState (entryBytes : Nat) (state : StoredState entryBytes) (side : Disc) :
    Bool × StoredState entryBytes :=
  let result := confirm entryBytes state.history.val state.history.property side
  (result.1,{state with history := ⟨result.2,(confirm_valid_frame entryBytes state.history.val
    state.history.property side).1⟩})

end NeuralReversi.Game.History.VecView

namespace NeuralReversi.Game.History
def entryBytes : Nat := 24
abbrev State := VecView.StoredState entryBytes

def undo (state : State) : Bool × State := VecView.undo entryBytes state

def confirmAutomaticPass (state : State) (side : Disc) : Bool × State :=
  VecView.confirmState entryBytes state side
end NeuralReversi.Game.History
