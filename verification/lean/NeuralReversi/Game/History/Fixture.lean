import NeuralReversi.Game.History.Model
import NeuralReversi.Simd.Fixture

namespace NeuralReversi.Game.History.Fixture
open NeuralReversi.Simd.X86 (parseWord)

def parseSide : String → Except String Disc
  | "0" => .ok .black
  | "1" => .ok .white
  | _ => .error "side outside typed domain"

def sideCode : Disc → Nat
  | .black => 0
  | .white => 1

def parseAction (text : String) : Except String Action := do
  let some value := text.toNat? | throw "invalid action"
  if bound : value < 64 then return .play ⟨value,bound⟩
  else if value = 65 then return .explicitPass
  else if value = 66 then return .automaticPass
  else throw "action outside typed domain"

def actionCode : Action → Nat
  | .play square => square.val
  | .explicitPass => 65
  | .automaticPass => 66

def parseEntry (text : String) : Except String Entry := do
  let [action,player,opponent,side] := text.splitOn "," | throw "expected action,player,opponent,side"
  return ⟨← parseAction action,⟨← parseWord player,← parseWord opponent⟩,← parseSide side⟩

def parseEntries (text : String) : Except String (List Entry) := do
  if text == "-" then return []
  let entries ← (text.splitOn ";").mapM parseEntry
  if entries.length > 128 then throw "entries exceed bounded comparison harness"
  return entries

def entryText (entry : Entry) : String :=
  s!"{actionCode entry.action},{entry.board.player.toNat},{entry.board.opponent.toNat},{sideCode entry.side}"

def entriesText (entries : List Entry) : String :=
  if entries.isEmpty then "-" else String.intercalate ";" (entries.map entryText)

def stateText (state : ValueState) : String :=
  s!"{state.board.player.toNat},{state.board.opponent.toNat},{sideCode state.side}/{entriesText state.history}"

def parseState (text : String) : Except String ValueState := do
  let [words,entries] := text.splitOn "/" | throw "expected state/history"
  let [player,opponent,side] := words.splitOn "," | throw "expected player,opponent,side"
  return ⟨⟨← parseWord player,← parseWord opponent⟩,← parseSide side,← parseEntries entries⟩

def valueRequest (fields : List String) : Except String String := do
  let [_,operation,before] := fields | throw "expected history operation before-state"
  let before ← parseState before
  let observed ← match operation with
    | "u" => pure (Reference.undo before)
    | "c0" => pure (Reference.confirm before .black)
    | "c1" => pure (Reference.confirm before .white)
    | _ => throw "unsupported history operation"
  return s!"{if observed.1 then 1 else 0}|{stateText observed.2}"

def validDecision (v : VecView.HistoryVec) : Decidable (VecView.Valid entryBytes v) := by
  unfold VecView.Valid VecView.AllocationValid entryBytes
  infer_instance

def metadata (v : VecView.HistoryVec) : String :=
  s!"{v.allocation.address.toNat},{v.allocation.capacityWord.toNat},{v.lengthWord.toNat}/{entriesText v.entries}"

def vecRequest (fields : List String) : Except String String := do
  let [_,operation,base,capacity,entries] := fields | throw "expected Vec operation base capacity entries"
  let base ← parseWord base
  let capacity ← parseWord capacity
  let entries ← parseEntries entries
  let v : VecView.HistoryVec := ⟨entries,BitVec.ofNat 64 entries.length,⟨0,base,capacity⟩⟩
  letI := validDecision v
  if valid : VecView.Valid entryBytes v then
    match operation with
    | "pop" =>
      let result := VecView.pop entryBytes v valid
      let value := result.1.map entryText |>.getD "-"
      return s!"{value}|{metadata result.2}"
    | "last" =>
      let value := match VecView.lastMut entryBytes v valid with
        | none => "-"
        | some borrow => s!"{entryText borrow.value}@{borrow.address.toNat}"
      return s!"{value}|{metadata v}"
    | _ => throw "unsupported Vec operation"
  else throw "invalid bounded Vec representation"

end NeuralReversi.Game.History.Fixture
