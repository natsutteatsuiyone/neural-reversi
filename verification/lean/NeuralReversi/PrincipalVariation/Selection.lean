module
public import NeuralReversi.PrincipalVariation.Buffer
public import NeuralReversi.Threading.SplitWindow

@[expose] public section

namespace NeuralReversi.PrincipalVariation

open Threading.SplitWindow (reSearch selectReport)

def serialChild (root first aborted : Bool) (alpha scout wide : Int)
    (previous completed : Buffer) : Option Buffer :=
  let widened := first || reSearch true alpha scout
  let score := if widened then -wide else -scout
  if !aborted && ((root && first) || decide (alpha < score)) then
    some (if widened then completed else previous)
  else none

def SerialContract : Prop :=
  ∀ (root first aborted : Bool) (alpha scout wide : Int) (previous completed carried : Buffer),
    (aborted = true → serialChild root first aborted alpha scout wide previous completed = none) ∧
    (serialChild root first aborted alpha scout wide previous completed = some carried →
      (first = true ∨ reSearch true alpha scout = true) ∧ carried = completed)

theorem serialContract_proven : SerialContract := by
  intro root first aborted alpha scout wide previous completed carried
  refine ⟨fun stopped => by simp [serialChild, stopped], ?_⟩
  intro got
  by_cases searched : (first || reSearch true alpha scout) = true
  · refine ⟨by simpa only [Bool.or_eq_true] using searched, ?_⟩
    unfold serialChild at got
    simp only [searched, ite_true] at got
    split at got
    · exact (Option.some.inj got).symm
    · cases got
  · have noSearch := (Bool.or_eq_false_iff.mp (Bool.eq_false_iff.mpr searched))
    have low : -scout ≤ alpha := by simpa [reSearch, Int.not_lt] using noSearch.2
    simp [serialChild, noSearch.1, noSearch.2, Int.not_lt_of_ge low] at got

def splitChild (aborted : Bool) (scoutAlpha freshAlpha lockAlpha scout wide : Int)
    (previous completed : Buffer) : Option Buffer :=
  let report := selectReport true 0 scoutAlpha freshAlpha scout wide
  if !aborted && decide (lockAlpha < report.score) then
    some (if reSearch true scoutAlpha scout then completed else previous)
  else none

def SplitContract : Prop :=
  ∀ (aborted : Bool) (scoutAlpha freshAlpha lockAlpha scout wide : Int) (previous completed carried : Buffer),
    scoutAlpha ≤ lockAlpha →
    (aborted = true → splitChild aborted scoutAlpha freshAlpha lockAlpha scout wide previous completed = none) ∧
    (splitChild aborted scoutAlpha freshAlpha lockAlpha scout wide previous completed = some carried →
      reSearch true scoutAlpha scout = true ∧ carried = completed)

theorem splitContract_proven : SplitContract := by
  intro aborted scoutAlpha freshAlpha lockAlpha scout wide previous completed carried ordered
  refine ⟨fun stopped => by simp [splitChild, stopped], ?_⟩
  intro got
  by_cases searched : reSearch true scoutAlpha scout = true
  · refine ⟨searched, ?_⟩
    unfold splitChild at got
    simp only [searched, ite_true] at got
    split at got
    · exact (Option.some.inj got).symm
    · cases got
  · have low : -scout ≤ scoutAlpha := by simpa [reSearch, Int.not_lt] using searched
    have locked : ¬ lockAlpha < -scout := Int.not_lt_of_ge (Int.le_trans low ordered)
    simp [splitChild, selectReport, searched, locked] at got

open Threading.AtomicWord (Read Write signed32At)
open Threading.CutoffVisibility (History)
open Threading.SplitWindow (AlphaRead Matches State scan)
open Aspiration (Window)

def LockedContract : Prop :=
  ∀ (history : History) (writes : List Write) (probe : Read writes) (first last : Fin writes.length)
    (original beta fresh scout wide : Int) (initial : State) (reports : List Int)
    (previous completed carried : Buffer) (aborted : Bool),
    AlphaRead history probe first last → Matches writes first last true beta initial reports →
    Window initial.alpha beta → initial.alpha ≤ max original initial.best →
    let observed := signed32At writes probe.source
    let lockAlpha := (scan true beta initial reports).alpha
    splitChild aborted observed fresh lockAlpha scout wide previous completed = some carried →
    reSearch true observed scout = true ∧ carried = completed

theorem lockedContract_proven : LockedContract := by
  intro history writes probe first last original beta fresh scout wide initial reports previous completed carried
    aborted read matched supported capped observed lockAlpha accepted
  have ordered := (Threading.SplitWindow.alphaReadContract_proven history writes probe first last true original beta
    initial reports read matched supported capped).2.1
  exact (splitContract_proven aborted observed fresh lockAlpha scout wide previous completed carried ordered).2 accepted

example : let old : Buffer := [some ⟨27, by decide⟩]
    serialChild true false false 0 0 (-10) old [] = none ∧
    serialChild true true false 0 0 5 old [] = some [] ∧
    splitChild false 3 5 7 (-4) (-6) old [] = none := by decide

end NeuralReversi.PrincipalVariation
