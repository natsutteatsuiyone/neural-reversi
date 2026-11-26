import NeuralReversi.EndgameSearch.Cancellation
import NeuralReversi.EndgameSearch.Thresholds

namespace NeuralReversi.EndgameSearch.DeepRecursionMir

open EndgameTreeCache (Address Store entryFor)
open Cancellation (EmptyState Outcome State)
open Bitboard.Moves (X86Config)
open PvsBoardTree (Position GeneratedSquare play)
open Flip.Dispatch (TargetConfig)

def stopped : Int := 0

def negate (r : Outcome) : Outcome := { r with score := r.score.map Neg.neg }

def executeChild (sq : Bitboard.Square) (beta : Int) (input : State)
    (shallow : EmptyState → Int → List Store → Result) (deep : Int → State → Outcome) : Outcome :=
  let polled := { input with polls := input.polls.after }
  if input.polls.occurred then ⟨none, polled⟩
  else
    let placed := { polled with empties := input.empties.remove sq }
    let out := if placed.empties.count.toNat ≤ 6 then
      let r := shallow placed.empties (-beta) placed.history
      ⟨some (-r.score), {placed with history := r.history}⟩
    else negate (deep (-beta) placed)
    { out with state := {out.state with empties := out.state.empties.restore sq} }

def finish (b : Position) (alpha best : Int) (state : State) : Outcome :=
  let polled := { state with polls := state.polls.after }
  if state.polls.occurred then ⟨none, polled⟩
  else ⟨some best, { polled with history := state.history ++ [⟨.deep, ⟨b.val, alpha, best⟩⟩] }⟩

def ordered {Move : Type} (child : Move → State → Outcome) (beta : Int) :
    List Move → Int → State → Int × State
  | [], best, state => (best, state)
  | m :: tail, best, state =>
    let out := child m state
    let score := out.score.getD stopped
    if beta ≤ score then (score, out.state) else ordered child beta tail (max best score) out.state

def early (fc : FlipConfig) (paired : Bool) (b : Position) (s : State) (addr : Address) (a : Int)
    (first : Bitboard.Square) (rest : List Bitboard.Square) : Option Int :=
  match EndgameCache.probe (entryFor addr s.history .deep b.val) b.val a with
  | some score => some score
  | none =>
    match EndgameStability.tryCutoff b.val s.empties.count.toNat ⟨a⟩ with
    | some score => some score.value
    | none =>
      let wipe := EndgameCutoffs.wipesOut fc b.val
      if rest.isEmpty then if wipe first then some 64 else none
      else
        let generated := MoveList.SmallGenerate.generate paired wipe (first :: rest)
        if generated.1 then
          if Thresholds.minEcEtcEmpties ≤ s.empties.count.toNat then
            let boards := generated.2.map fun sq => Board.place b.val
              (Flip.Dispatch.resultValue fc sq b.val.player b.val.opponent) sq
            EndgameCache.enhancedCutoff (entryFor addr s.history .deep) a boards
          else none
        else some 64

def moves (b : Position) (a : Int) (state : State) (first : Move) (rest orderedMoves : List Move)
    (child : Move → State → Outcome) : Outcome :=
  let result := if rest.isEmpty then
    let r := child first state
    (r.score.getD stopped, r.state)
  else ordered child (a + 1) orderedMoves (-127) state
  finish b a result.1 result.2

end NeuralReversi.EndgameSearch.DeepRecursionMir
