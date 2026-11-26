import NeuralReversi.PvsParallel.RootAspiration

namespace NeuralReversi.PvsParallel.RootMultiPV

open Aspiration (Window)
open Bitboard (Square)
open Bitboard.Moves (X86Config)
open PvsBoardTree (Position)
open RootAspiration (Pick SoundSearch runLine)

abbrev Observations := List Square → Nat → RootAspiration.Search
abbrev Seed := List Square → Nat → Int × Int

structure Batch where
  picks : List Pick
  order : List Square

def run (observe : Observations) (seed : Seed) : Nat → Nat → List Square → Option Batch
  | 0, _, roots => some ⟨[], roots⟩
  | count + 1, idx, roots =>
    let (alpha, beta) := seed roots idx
    (runLine (observe roots idx) roots idx alpha beta).bind fun line =>
      (run observe seed count (idx + 1) line.order).map fun rest => ⟨line.pick :: rest.picks, rest.order⟩

def Ranked (values : Square → Int) : List Square → List Pick → Prop
  | _, [] => True
  | candidates, pick :: rest =>
    pick.square ∈ candidates ∧ pick.score = values pick.square ∧
      (∀ square ∈ candidates, values square ≤ pick.score) ∧ Ranked values (candidates.erase pick.square) rest

private theorem ranked_perm (order : left.Perm right) (ranking : Ranked values left picks) :
    Ranked values right picks := by
  induction picks generalizing left right with
  | nil => exact True.intro
  | cons pick rest ih =>
    obtain ⟨member, exactScore, best, next⟩ := ranking
    exact ⟨order.mem_iff.mp member, exactScore, fun sq mem => best sq (order.mem_iff.mpr mem),
      ih (order.erase pick.square) next⟩

private theorem ranked_subset (ranking : Ranked values candidates picks) :
    ∀ pick ∈ picks, pick.square ∈ candidates ∧ pick.score = values pick.square := by
  induction picks generalizing candidates with
  | nil => simp
  | cons first rest ih =>
    intro pick member
    obtain ⟨legal, exactScore, _, next⟩ := ranking
    rcases List.mem_cons.mp member with rfl | later
    · exact ⟨legal, exactScore⟩
    · obtain ⟨kept, value⟩ := ih next pick later
      exact ⟨List.mem_of_mem_erase kept, value⟩

theorem Ranked.ordered (ranking : Ranked values candidates picks) :
    picks.Pairwise (fun a b => b.score ≤ a.score) := by
  induction picks generalizing candidates with
  | nil => exact .nil
  | cons first rest ih =>
    obtain ⟨_, _, maximal, next⟩ := ranking
    refine .cons (fun later member => ?_) (ih next)
    obtain ⟨kept, score⟩ := ranked_subset next later member
    rw [score]
    exact maximal later.square (List.mem_of_mem_erase kept)

private theorem ranked_distinct (distinct : candidates.Nodup) (ranking : Ranked values candidates picks) :
    (picks.map Pick.square).Nodup := by
  induction picks generalizing candidates with
  | nil => exact List.nodup_nil
  | cons first rest ih =>
    have residual : candidates.Perm (first.square :: candidates.erase first.square) :=
      List.perm_cons_erase ranking.1
    have unique := residual.nodup_iff.mp distinct
    refine List.nodup_cons.mpr ⟨?_, ih (List.nodup_cons.mp unique).2 ranking.2.2.2⟩
    intro twice
    obtain ⟨later, member, same⟩ := List.mem_map.mp twice
    have kept := (ranked_subset ranking.2.2.2 later member).1
    exact (List.nodup_cons.mp unique).1 (same ▸ kept)

private theorem ranked_all (ranking : Ranked values candidates picks) (count : picks.length = candidates.length) :
    (picks.map Pick.square).Perm candidates := by
  induction picks generalizing candidates with
  | nil =>
    have empty : candidates = [] := List.length_eq_zero_iff.mp count.symm
    rw [empty]
    exact List.Perm.nil
  | cons first rest ih =>
    have order := List.perm_cons_erase ranking.1
    have len := order.length_eq
    have left : rest.length = (candidates.erase first.square).length := by
      simp only [List.length_cons] at count len
      omega
    exact (List.Perm.cons _ (ih ranking.2.2.2 left)).trans order.symm

private theorem retain_head {later earlier : List Square} (frozen : later.take (idx + 1) = earlier.take (idx + 1))
    (head : (earlier.drop idx).head? = some square) :
    later.drop idx = square :: later.drop (idx + 1) := by
  have keep := congrArg (fun (roots : List Square) => roots[idx]?) frozen
  rw [List.getElem?_take_of_succ, List.getElem?_take_of_succ] at keep
  have chosen : (later.drop idx).head? = some square := by
    rw [List.head?_drop, keep, ← List.head?_drop]
    exact head
  rw [List.drop_add_one_eq_tail_drop]
  cases found : later.drop idx with
  | nil => simp [found] at chosen
  | cons first rest =>
    have same : first = square := Option.some.inj (by simpa only [found, List.head?_cons] using chosen)
    subst first
    rfl

def Matched (mc : X86Config) (fc : Root.FlipConfig) (board : Position) (observe : Observations) (seed : Seed) :
    Prop :=
  ∀ roots idx, Root.Covers mc board roots → idx < roots.length →
    Window (seed roots idx).1 (seed roots idx).2 ∧
      SoundSearch mc fc board roots idx (observe roots idx) ∧
      ∀ i a b, Window a b → (observe roots idx i a b).aborted = false

def LineCalls (observe : Observations) (seed : Seed)
    (checked : List Square → Nat → RootAspiration.Line → Prop) : Nat → Nat → List Square → Prop
  | 0, _, _ => True
  | count + 1, idx, roots =>
    ∃ line, runLine (observe roots idx) roots idx (seed roots idx).1 (seed roots idx).2 = some line ∧
      checked roots idx line ∧ LineCalls observe seed checked count (idx + 1) line.order

private theorem LineCalls.ofMatched (matched : Matched mc fc board observe seed)
    (covered : Root.Covers mc board roots) (enough : idx + count ≤ roots.length) :
    LineCalls observe seed (RootAspiration.LineSound (Root.squareValue mc fc board)) count idx roots := by
  induction count generalizing roots idx with
  | zero => trivial
  | succ count ih =>
    have inside : idx < roots.length := by omega
    obtain ⟨window, sound, uninterrupted⟩ := matched roots idx covered inside
    obtain ⟨line, _, _, _, found, _, _, valid⟩ := RootAspiration.lineContract_proven
      mc fc board roots idx (seed roots idx).1 (seed roots idx).2 (observe roots idx)
      covered inside window sound uninterrupted
    refine ⟨line, found, valid, ih (valid.order.trans covered) ?_⟩
    rw [valid.order.length_eq]
    omega

theorem LineCalls.ranked {checked : List Square → Nat → RootAspiration.Line → Prop}
    (calls : LineCalls observe seed checked count idx roots)
    (sound : ∀ order j line, Root.Covers mc board order → j < order.length →
      runLine (observe order j) order j (seed order j).1 (seed order j).2 = some line →
      checked order j line → RootAspiration.LineSound (Root.squareValue mc fc board) order j line)
    (covered : Root.Covers mc board roots) (enough : idx + count ≤ roots.length) :
    ∃ result, run observe seed count idx roots = some result ∧
      result.picks.length = count ∧
      result.order.Perm roots ∧ result.order.take idx = roots.take idx ∧
      (result.order.drop idx).take count = result.picks.map Pick.square ∧
      Ranked (Root.squareValue mc fc board) (roots.drop idx) result.picks := by
  induction count generalizing roots idx with
  | zero => exact ⟨⟨[], roots⟩, rfl, rfl, List.Perm.refl _, rfl, rfl, True.intro⟩
  | succ count ih =>
    have inside : idx < roots.length := by omega
    obtain ⟨line, lineFound, checkedLine, restCalls⟩ := calls
    have lineSound := sound roots idx line covered inside lineFound checkedLine
    have capacity : idx + 1 + count ≤ line.order.length := by
      rw [lineSound.order.length_eq]
      omega
    obtain ⟨rest, found, length, order, fixed, picked, ranked⟩ :=
      ih restCalls (lineSound.order.trans covered) capacity
    refine ⟨⟨line.pick :: rest.picks, rest.order⟩, ?_, ?_, order.trans lineSound.order, ?_, ?_, ?_⟩
    · simp only [run, lineFound, Option.bind_some, found, Option.map_some]
    · simp only [List.length_cons, length]
    · change rest.order.take idx = roots.take idx
      have saved := congrArg (List.take idx) fixed
      simp only [List.take_take, Nat.min_eq_left (Nat.le_succ _)] at saved
      exact saved.trans lineSound.fixed
    · change (rest.order.drop idx).take (count + 1) = (line.pick :: rest.picks).map Pick.square
      rw [retain_head fixed lineSound.chosen, List.take_succ_cons, List.map_cons, picked]
    · exact ⟨lineSound.legal, lineSound.exactScore, lineSound.maximal, ranked_perm lineSound.remaining ranked⟩

def Contract : Prop :=
  ∀ (mc : X86Config) (fc : Root.FlipConfig) (board : Position) (roots : List Square)
    (count : Nat) (observe : Observations) (seed : Seed),
    Root.Covers mc board roots → count ≤ roots.length → Matched mc fc board observe seed →
    ∃ result, run observe seed count 0 roots = some result ∧
      result.picks.length = count ∧
      result.order.Perm roots ∧ result.order.take count = result.picks.map Pick.square ∧
      Ranked (Root.squareValue mc fc board) roots result.picks ∧
      result.picks.Pairwise (fun a b => b.score ≤ a.score) ∧ (result.picks.map Pick.square).Nodup ∧
      (count = roots.length → (result.picks.map Pick.square).Perm roots)

theorem contract_proven : Contract := by
  intro mc fc board roots count observe seed covered enough matched
  have capacity : 0 + count ≤ roots.length := by simpa using enough
  obtain ⟨result, found, length, order, _, chosen, ranked⟩ :=
    (LineCalls.ofMatched matched covered capacity).ranked (fun _ _ _ _ _ _ valid => valid) covered capacity
  exact ⟨result, found, length, order, chosen, ranked, ranked.ordered,
    ranked_distinct (Root.roots_nodup covered) ranked, fun all => ranked_all ranked (length.trans all)⟩

private def mc : X86Config := ⟨false, false⟩
private def fc : Root.FlipConfig := ⟨false, false, false⟩

private def observeTwoOpen : Observations := fun roots idx _ alpha beta =>
  let board := RootReplay.twoOpen
  let moves := Root.choices mc board roots idx
  let observe := fun m => PvsMoveLoop.run (PvsBoardTree.build mc fc (PvsBoardTree.play mc fc board m).val)
  let attempt := RootReplay.serial Subtype.val observe beta moves (RootReplay.start alpha)
  ⟨attempt.state.best, attempt.scores, false⟩

private def narrowSeeds : Seed := fun _ idx => if idx = 0 then (0, 256) else (-7680, -7424)

private theorem accepts_second (search : RootAspiration.Search) (alpha beta a b : Int)
    (notAborted : (search 0 alpha beta).aborted = false) (stillRunning : (search 1 a b).aborted = false)
    (retry : Aspiration.widen (search 0 alpha beta).score alpha beta 256 = some (a, b))
    (accept : Aspiration.widen (search 1 a b).score a b 512 = none) :
    Aspiration.runWith search 11 0 alpha beta 256 = some ⟨1, a, b, search 1 a b⟩ := by
  rw [Aspiration.runWith]
  simp only [notAborted, Bool.false_eq_true, ite_false, retry]
  rw [Aspiration.runWith]
  change (if (search 1 a b).aborted then none
      else match Aspiration.widen (search 1 a b).score a b 512 with
      | none => some ⟨1, a, b, search 1 a b⟩
      | some (lo, hi) => Aspiration.runWith search 9 2 lo hi 1024) = _
  simp only [stillRunning, Bool.false_eq_true, ite_false, accept]

example :
    (run observeTwoOpen narrowSeeds 2 0 [14, 47]).map
      (fun batch => batch.picks.map fun pick => (pick.square, pick.score)) = some [(47, -1024), (14, -4096)] := by
  have acceptedFirst := accepts_second (observeTwoOpen [14, 47] 0) 0 256 (-1280) 0
    rfl rfl (by decide +kernel) (by decide +kernel)
  have acceptedSecond := accepts_second (observeTwoOpen [47, 14] 1) (-7680) (-7424) (-7680) (-3840)
    rfl rfl (by decide +kernel) (by decide +kernel)
  let report := observeTwoOpen [14, 47] 0 1 (-1280) 0
  have weaker : report.payload 14 = -4096 := by decide +kernel
  have stronger : report.payload 47 = -1024 := by decide +kernel
  have reordered : RootOrder.sortFrom [14, 47] 0 report.payload = [47, 14] := by
    change ([14, 47] : List Square).mergeSort _ = _
    rw [List.mergeSort_pair]
    simp [weaker, stronger]
  have first : runLine (observeTwoOpen [14, 47] 0) [14, 47] 0 0 256 =
      some ⟨⟨47, -1024⟩, [47, 14]⟩ := by
    rw [RootAspiration.runLine, acceptedFirst]
    simp only [Option.bind_some, RootAspiration.finish]
    change (match (RootOrder.sortFrom [14, 47] 0 report.payload).drop 0 with
      | [] => none
      | head :: _ => some (⟨⟨head, report.score⟩,
          RootOrder.sortFrom [14, 47] 0 report.payload⟩ : RootAspiration.Line)) = _
    rw [reordered]
    have score : report.score = -1024 := by decide +kernel
    rw [score]
    rfl
  have second : runLine (observeTwoOpen [47, 14] 1) [47, 14] 1 (-7680) (-7424) =
      some ⟨⟨14, -4096⟩, [47, 14]⟩ := by
    rw [RootAspiration.runLine, acceptedSecond]
    simp only [Option.bind_some, RootAspiration.finish, RootOrder.sortFrom, List.drop_succ_cons,
      List.drop_zero, List.mergeSort_singleton]
    have score : (observeTwoOpen [47, 14] 1 1 (-7680) (-3840)).score = -4096 := by decide +kernel
    rw [score]
    rfl
  simp only [run, narrowSeeds, ite_true, first, Option.bind_some, Option.map_some,
    Nat.zero_add, Nat.one_ne_zero, ite_false, second]
  rfl

end NeuralReversi.PvsParallel.RootMultiPV
