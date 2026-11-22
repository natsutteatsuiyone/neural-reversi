import NeuralReversi.Game.History.Spec

namespace NeuralReversi.Game.History.Reference

def popLast : List Entry → Option (Entry × List Entry)
  | [] => none
  | [entry] => some (entry,[])
  | head::next::tail => (popLast (next::tail)).map fun (entry,rest) => (entry,head::rest)

def undo (state : ValueState) : Bool × ValueState :=
  match popLast state.history with
  | none => (false,state)
  | some (entry,rest) =>
    match entry.action with
    | .automaticPass =>
      match popLast rest with
      | none => (true,restore state entry rest)
      | some (before,initial) => (true,restore state before initial)
    | .play _ | .explicitPass => (true,restore state entry rest)

def confirm (state : ValueState) (side : Disc) : Bool × ValueState :=
  match popLast state.history with
  | none => (false,state)
  | some (entry,rest) =>
    if entry.action = .automaticPass ∧ entry.side = side then
      (true,{state with history := rest ++ [{entry with action := .explicitPass}]})
    else (false,state)

end NeuralReversi.Game.History.Reference
