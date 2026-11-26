import NeuralReversi.AspirationMir.SerialCache.Driver
import NeuralReversi.PublicationMir.ExactScore

namespace NeuralReversi.AspirationMir.SerialTable

open PvsParallel
open PvsBoardTree (Position)
open RootMoveStorage (Row)
open SerialStack (State Certified)
open SerialCache (Receipt Trial Path)

def Works (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (trial : Trial M) (good : M → Prop) (evidence : Receipt M → Prop) : Prop :=
  ∀ s before, good before → Root.Covers mc b (s.rows.map Row.square) → 0 < s.rows.length →
    Aspiration.Window s.regs.alpha.toInt s.regs.beta.toInt →
    ∃ closing answer after,
      trial s before = some (closing,answer,after) ∧ Certified mc fc b s answer ∧ good after ∧ s.opening ≤ closing ∧
      evidence ⟨⟨s,closing,answer⟩,before,after⟩

theorem opening_chain {trial : Trial M} (path : Path trial s before prior last) :
    (prior ++ [last]).map (fun v => v.state.opening) = s.opening :: prior.map (fun v => v.closing + 1) := by
  induction path with
  | accept => rfl
  | retry _ _ _ ih => exact congrArg (_ :: ·) ih

def Published (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (trial : Trial M) (good : M → Prop) (evidence : Receipt M → Prop)
    (rows : List Row) (m : Machine) (before : M)
    (sort : PublicationMir.SliceSort) (nodes : Nat) (fallback : Unit → PublicationMir.SearchResult) : Prop :=
  ∃ prior last result,
    SerialCache.run trial 12 (SerialStack.first m rows) before = some (prior,last) ∧
    Path trial (SerialStack.first m rows) before prior last ∧ prior.length < 12 ∧
    (prior ++ [last]).map Receipt.before = before :: prior.map Receipt.after ∧
    (prior ++ [last]).map (fun v => v.state.opening) = 0 :: prior.map (fun v => v.closing + 1) ∧
    ((prior ++ [last]).map Receipt.toVisit).flatMap SerialStack.Visit.openings = List.range (last.closing + 1) ∧
    good last.after ∧
    (∀ v ∈ prior ++ [last], good v.before ∧ good v.after ∧ Certified mc fc b v.state v.answer ∧ v.state.opening ≤ v.closing ∧ evidence v) ∧
    Returns (SerialStack.queryFor ((prior ++ [last]).map Receipt.toVisit)) (fun _ => false) m last.call.score last.call ∧
    PublicationMir.completed sort (RootMoveStorage.sortRows last.answer.rows 0)
      (Board.emptyCount b.val).toNat 5 true nodes fallback = some result ∧
    PublicationMir.ExactScore.scoreIfExact result = some (256 * EndgameTreeCache.discValue mc fc b.val) ∧
    last.answer.value = PvsBoardTree.minimax mc fc b

theorem publication (trial : Trial M) (works : Works mc fc b trial good evidence)
    (initial : good before) (cover : Root.Covers mc b (rows.map Row.square)) (positive : 0 < rows.length)
    (m : Machine) (pc : m.pc = 0) (window : Aspiration.Window m.regs.alpha.toInt m.regs.beta.toInt)
    (sort : PublicationMir.SliceSort) (nodes : Nat) (fallback : Unit → PublicationMir.SearchResult) :
    Published mc fc b trial good evidence rows m before sort nodes fallback := by
  let seen (v : Receipt M) := good v.before ∧ good v.after ∧ Certified mc fc b v.state v.answer ∧ v.state.opening ≤ v.closing ∧ evidence v
  obtain ⟨prior,last,ran,path,bound,covered,nonempty,cert,inside,_,safe,observed⟩ :=
    SerialCache.completeState (trial := trial) (valid := good) (evidence := seen)
      (extendsState := fun _ _ => True) (fun _ _ => True.intro)
      (fun s h hGood names hasMoves w => by
        obtain ⟨k,ans,after,run,cert,hAfter,clock,ev⟩ := works s h hGood names hasMoves w
        exact ⟨k,ans,after,run,cert,hAfter,True.intro,hGood,hAfter,cert,clock,ev⟩)
      12 (SerialStack.first m rows) before initial (first_ready _ window) rfl cover positive
  obtain ⟨snap,best,writes,att⟩ := cert
  obtain ⟨result,published,score⟩ := publishSingle covered nonempty writes (best ▸ inside) (best ▸ att)
    (PublicationMir.ExactScore.afterRootSort sort) (Board.emptyCount b.val).toNat nodes fallback
  obtain ⟨_,chain⟩ := path.trials
  have corresponding : (prior ++ [last]).map Receipt.toVisit = prior.map Receipt.toVisit ++ [last.toVisit] := by simp
  have intervals := (chain.openings (fun v mem => by
    rw [← corresponding] at mem
    obtain ⟨receipt,belongs,rfl⟩ := List.mem_map.mp mem
    exact (observed receipt belongs).2.2.2.1)).1
  refine ⟨prior,last,result,ran,path,bound,path.carried,opening_chain path,?_,safe,observed,
    path.returns m pc,published,?_,?_⟩
  · simpa only [corresponding,SerialStack.first,Nat.sub_zero,List.range_eq_range'] using intervals
  · rw [PublicationMir.ExactScore.completed_tagged published,EndgameTreeCache.scaled_discValue]
    exact score
  · exact (att.1.exact inside.1 inside.2).trans (Root.value_zero covered (by simpa using nonempty))

theorem started_of_certificate (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (paired : Bool) (mem : MoveList.Initialization.Memory Move.Move)
    (checked : Std.Tactic.BVDecide.LRAT.check certificate (LegalMoves.MoveBound.dimacsFormula 35) = true)
    (mobile : Board.hasLegalMoves mc b.val = true) :
    NativeRootBirth.started mc fc b paired mem = some (NativeRootBirth.initial mc b) := by
  have nonempty : PvsBoardTree.legalSquares mc b.val ≠ [] :=
    fun none => Bool.noConfusion ((PvsBoardTree.stuck_of_nil none).symm.trans mobile)
  simp only [NativeRootBirth.started,NativeRootBirth.certified mc fc b paired mem checked]
  cases order : PvsBoardTree.legalSquares mc b.val with
  | nil => contradiction
  | cons _ _ => simp [NativeRootBirth.initial,order]

end NeuralReversi.AspirationMir.SerialTable
