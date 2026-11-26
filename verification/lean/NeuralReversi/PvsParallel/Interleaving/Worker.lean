import NeuralReversi.PvsParallel.SplitWorkerMir.Loop

namespace NeuralReversi.PvsParallel.Interleaving.Worker

open Threading
open SplitWorkerMir (Input Searched Locked Commit Word issued retry selected scoutQuery wideQuery)
open Aspiration (Window SoundResult)
open PvsMoveLoop (Bound scoreMax)

def report (i : Input) : SplitResult.Report := selected i (issued i)

def Children (v : Int) (i : Input) : Prop :=
  ∀ q, .returned q ∈ (issued i).steps → Window q.alpha.toInt q.beta.toInt →
    SoundResult (-v) q.alpha.toInt q.beta.toInt q.result.toInt

theorem search_returns (i : Input) (depth : 0 < i.depth) (suffix : i.pv = true → 0 < i.index) :
    SplitWorkerMir.search i = some (issued i) := by
  rw [SplitWorkerMir.search_eq,ite_eq_right (by intro bad; rcases bad with d | ⟨pv,j⟩; omega; have := suffix pv; omega)]

private theorem scout_operands (i : Input) (w : Window i.alpha.toInt i.beta.toInt) :
    (scoutQuery i).alpha.toInt = -(i.alpha.toInt+1) ∧ (scoutQuery i).beta.toInt = -i.alpha.toInt := by
  have lo := w.lower
  have lt := w.ordered
  have hi := w.upper
  exact ⟨Solve2.childAlpha_safe _ (by omega), Solve2.neg_safe _ (by omega)⟩

private theorem wide_operands (i : Input) (w : Window i.fresh.toInt i.beta.toInt) :
    (wideQuery i).alpha.toInt = -i.beta.toInt ∧ (wideQuery i).beta.toInt = -i.fresh.toInt := by
  have lo := w.lower
  have lt := w.ordered
  exact ⟨Solve2.neg_safe _ (by omega), Solve2.neg_safe _ (by omega)⟩

theorem sound (i : Input) (locked : Int) (window : Window i.alpha.toInt i.beta.toInt)
    (sample : i.alpha.toInt ≤ locked) (narrow : i.pv = false → i.beta.toInt = i.alpha.toInt+1)
    (reread : retry i = true → Window i.fresh.toInt i.beta.toInt ∧ i.fresh.toInt ≤ locked)
    (children : Children v i) :
    Bound v (report i).searchedAlpha i.beta.toInt (report i).score ∧ (report i).searchedAlpha ≤ locked := by
  obtain ⟨a,b⟩ := scout_operands i window
  have scout := children (scoutQuery i) (by unfold issued; split <;> simp)
  rw [a,b] at scout
  have sc := scout (PvsMoveLoop.scoutWindow window)
  have sw : -32767 < i.scout.toInt := sc.lower
  have retryEq : retry i = SplitWindow.reSearch i.pv i.alpha.toInt i.scout.toInt := by
    simp [retry, SplitWindow.reSearch, Solve2.neg_safe i.scout (by omega)]
  have wide (yes : retry i = true) := children (wideQuery i) (by simp [issued,yes])
  have wideSound (yes : retry i = true) : SoundResult (-v) (-i.beta.toInt) (-i.fresh.toInt) i.wide.toInt := by
    obtain ⟨x,y⟩ := wide_operands i (reread yes).1
    have s := wide yes
    rw [x,y] at s
    exact s (PvsMoveLoop.reflectedWindow (reread yes).1)
  have good := SplitWindow.childReport (value := v) (scout := i.scout.toInt) (wide := i.wide.toInt) i.index
    window sample narrow (fun r => reread (retryEq.trans r)) (fun _ => sc)
    (fun r _ => wideSound (retryEq.trans r))
  have same : report i = SplitWindow.selectReport i.pv i.index i.alpha.toInt i.fresh.toInt i.scout.toInt i.wide.toInt := by
    cases r : retry i with
    | false =>
      simp [report,selected,issued,r,SplitWindow.selectReport,←retryEq,Solve2.neg_safe i.scout (by omega)]
    | true =>
      have w : -32767 < i.wide.toInt := (wideSound r).lower
      simp [report,selected,issued,r,SplitWindow.selectReport,←retryEq,Solve2.neg_safe i.wide (by omega)]
  simpa only [same] using good

theorem publish (i : Input) (l : Locked) (run : l.stopped = false) :
    SplitWorkerMir.lockedState l (SplitWorkerMir.commit i l (issued i).chosen) =
      SplitWindow.publish i.pv i.beta.toInt ⟨l.alpha.toInt,l.best.toInt⟩ (report i).score :=
  SplitWorkerMir.publish_eq i l _ run

theorem marks (i : Input) (l : Locked) (run : l.stopped = false) :
    (SplitWorkerMir.commit i l (issued i).chosen).cut =
      SplitWindow.marksCutoff i.pv i.beta.toInt ⟨l.alpha.toInt,l.best.toInt⟩ (report i).score :=
  SplitWorkerMir.cutoff_flag i l _ run

theorem cutoff (i : Input) (l : Locked) (run : l.stopped = false)
    (narrow : i.pv = false → i.beta.toInt = l.alpha.toInt+1)
    (cut : (SplitWorkerMir.commit i l (issued i).chosen).cut = true) :
    i.beta.toInt ≤ (report i).score ∨ scoreMax ≤ (report i).score := by
  rw [marks i l run] at cut
  obtain ⟨_,improves,why⟩ := of_decide_eq_true cut
  rcases why with notPV | high | maxed
  · have eq := narrow notPV
    exact Or.inl (by dsimp only at improves; omega)
  · exact Or.inl high
  · exact Or.inr maxed

theorem discarded (i : Input) (l : Locked) (stop : l.stopped = true) :
    SplitWorkerMir.commit i l (issued i).chosen = ⟨.stopped,none,none,false,none,false⟩ := by
  rw [SplitWorkerMir.commit_eq,stop]
  rfl

/-- The scout chooses whether the second alpha load and the wide child are needed. -/
def issue (pv root cut : Bool) (depth index : Nat) (alpha beta fresh : Int) (child : PvsMoveLoop.ChildSearch Nat) : Input :=
  let scout := child index false (-(alpha+1)) (-alpha)
  let wide := if SplitWindow.reSearch pv alpha scout then child index true (-beta) (-fresh) else 0
  ⟨index,depth,pv,root,cut,BitVec.ofInt 32 beta,BitVec.ofInt 32 alpha,BitVec.ofInt 32 fresh,
    BitVec.ofInt 32 scout,BitVec.ofInt 32 wide⟩

structure Issued (v : Int) (i : Input) (alpha beta fresh scout wide : Int) : Prop where
  alphaWord : i.alpha.toInt = alpha
  betaWord : i.beta.toInt = beta
  retried : retry i = SplitWindow.reSearch i.pv alpha scout
  freshWord : retry i = true → i.fresh.toInt = fresh
  children : Children v i
  selected : report i = SplitWindow.selectReport i.pv i.index alpha fresh scout wide

private theorem word_eq {x : Int} (lo : -32767 ≤ x) (hi : x ≤ 32767) :
    (BitVec.ofInt 32 x).toInt = x :=
  BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)

theorem issue_spec (pv root cut : Bool) (depth index : Nat) (alpha beta fresh : Int)
    (child : PvsMoveLoop.ChildSearch Nat) (w : Window alpha beta)
    (freshWindow : SplitWindow.reSearch pv alpha (child index false (-(alpha+1)) (-alpha)) = true → Window fresh beta)
    (scoutSound : Window (-(alpha+1)) (-alpha) →
      SoundResult (-v) (-(alpha+1)) (-alpha) (child index false (-(alpha+1)) (-alpha)))
    (wideSound : SplitWindow.reSearch pv alpha (child index false (-(alpha+1)) (-alpha)) = true →
      Window (-beta) (-fresh) → SoundResult (-v) (-beta) (-fresh) (child index true (-beta) (-fresh))) :
    Issued v (issue pv root cut depth index alpha beta fresh child) alpha beta fresh
      (child index false (-(alpha+1)) (-alpha)) (child index true (-beta) (-fresh)) := by
  let i := issue pv root cut depth index alpha beta fresh child
  let scout := child index false (-(alpha+1)) (-alpha)
  let wide := child index true (-beta) (-fresh)
  let again := SplitWindow.reSearch pv alpha scout
  have lower := w.lower
  have upper := w.upper
  have ord := w.ordered
  have aw : i.alpha.toInt = alpha := word_eq lower (by omega)
  have bw : i.beta.toInt = beta := word_eq (by omega) upper
  have sc := scoutSound (PvsMoveLoop.scoutWindow w)
  have scLo : -32767 < scout := sc.lower
  have scHi : scout < 32767 := sc.upper
  have sw : i.scout.toInt = scout := word_eq (by omega) (by omega)
  have ns : (-i.scout).toInt = -scout := (Solve2.neg_safe _ (by omega)).trans (congrArg Neg.neg sw)
  have tag : i.pv = pv := rfl
  have same : retry i = again := by simp [retry,again,SplitWindow.reSearch,tag,aw,ns]
  have fw (r : again = true) : i.fresh.toInt = fresh := by
    have win := freshWindow r
    have := win.ordered
    exact word_eq win.lower (by omega)
  have ww (r : again = true) : i.wide.toInt = wide := by
    have wideOk := wideSound r (PvsMoveLoop.reflectedWindow (freshWindow r))
    have dlo : -32767 < wide := wideOk.lower
    have dhi : wide < 32767 := wideOk.upper
    change (BitVec.ofInt 32 (if again then wide else 0)).toInt = wide
    rw [r,ite_eq_left rfl]
    exact word_eq (by omega) (by omega)
  have opw : Window i.alpha.toInt i.beta.toInt := by rwa [aw,bw]
  have scoutOK : SoundResult (-v) (scoutQuery i).alpha.toInt (scoutQuery i).beta.toInt (scoutQuery i).result.toInt := by
    obtain ⟨a,b⟩ := scout_operands i opw
    rw [a,b,aw]
    change SoundResult (-v) (-(alpha+1)) (-alpha) i.scout.toInt
    rwa [sw]
  have wideOK (r : again = true) :
      SoundResult (-v) (wideQuery i).alpha.toInt (wideQuery i).beta.toInt (wideQuery i).result.toInt := by
    have openWindow : Window i.fresh.toInt i.beta.toInt := by rw [fw r,bw]; exact freshWindow r
    obtain ⟨a,b⟩ := wide_operands i openWindow
    rw [a,b,fw r,bw]
    change SoundResult (-v) (-beta) (-fresh) i.wide.toInt
    rw [ww r]
    exact wideSound r (PvsMoveLoop.reflectedWindow (freshWindow r))
  change Issued v i alpha beta fresh scout wide
  refine ⟨aw,bw,tag ▸ same,(fun r => fw (same.symm.trans r)),?_,?_⟩
  · intro q member _
    cases r : retry i with
    | false =>
      simp [issued,r] at member
      exact member ▸ scoutOK
    | true =>
      simp [issued,r] at member
      rcases member with eq | eq
      · exact eq ▸ scoutOK
      · exact eq ▸ wideOK (same.symm.trans r)
  · cases r : retry i with
    | false =>
      have no : SplitWindow.reSearch i.pv alpha scout = false := by rw [tag]; exact same.symm.trans r
      simp [report,SplitWindow.selectReport,issued,r,no,SplitWorkerMir.selected,ns,aw]
    | true =>
      have yes : again = true := same.symm.trans r
      have rerun : SplitWindow.reSearch i.pv alpha scout = true := by rw [tag]; exact yes
      have lo : -32767 < wide := (wideSound yes (PvsMoveLoop.reflectedWindow (freshWindow yes))).lower
      have neg : (-i.wide).toInt = -wide :=
        (Solve2.neg_safe _ (by have := ww yes; omega)).trans (congrArg Neg.neg (ww yes))
      simp [report,SplitWindow.selectReport,issued,r,rerun,SplitWorkerMir.selected,neg,fw yes]

end NeuralReversi.PvsParallel.Interleaving.Worker
