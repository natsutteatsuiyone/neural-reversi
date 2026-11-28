module
public import NeuralReversi.PvsParallel.SerialTable.MidgameShallow
public import NeuralReversi.PvsParallel.SerialTable.MidgameSites
public import NeuralReversi.PvsParallel.SerialTable.MidgameNativeWords

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialTable.MidgameNative

open SerialCalls (Request)
open MidgameChild (Plan)
open MidgameValue (Value Sample known call)
open PvsBoardTree (Position GeneratedSquare legalSquares play pass)
open Bitboard (Square)
open TranspositionTable (EntryData Selectivity Depth)
open EndgameSearch.Cancellation (EmptyState)
open EndgameSearch.SerialLeafCache (Call)
open EndgameSearch.CacheHistory (Pool)
open AspirationMir.SerialTable.Repeated (ReadyMemory)

abbrev Path := List Nat

/-- Inputs visible to the search but immaterial to the endgame score. A path identifies
    the call site and move, including two visits to the same board during a retry. -/
structure Heuristics where
  shallow : Path → MidgameShallow.Inputs
  ordering : Path → Request → Option EntryData → Square → Int
  tie : Path → Request → Square → Nat
  estimates : Path → Request → Int × Int × Int
  stop : Path → Request → Square → Bool := fun _ _ _ => false

structure Inputs extends Heuristics where
  config : SerialScout.Config
  level : Selectivity

abbrev Empties (b : Position) := {s : EmptyState // s.matches b}

def initial (b : Position) : Empties b :=
  ⟨ContextPathMir.leafState b,EndgameSearch.emptyState_initial b⟩

def placeState (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (s : Empties b) (move : GeneratedSquare mc b.val) : Empties (play mc fc b move).val :=
  ⟨s.val.remove move.val,(EndgameSearch.emptyStatePlayContract_proven mc fc b move _ _ s.property).2.1⟩

/-- One same-depth forced pass is possible between reductions in depth.
    Placement, IIR before placement, and ProbCut verification all reduce depth. -/
def rank (mc : Bitboard.Moves.X86Config) (q : Request) : Nat :=
  q.depth.toNat * 2 + if Board.hasLegalMoves mc q.board.val then 0 else 1

theorem lower_rank (mc : Bitboard.Moves.X86Config) (q r : Request) (h : r.depth.toNat < q.depth.toNat) :
    rank mc r < rank mc q := by
  unfold rank
  split <;> split <;> omega

private theorem sub_one (d : Depth) (h : 0 < d.toNat) : (d-1).toNat = d.toNat-1 :=
  BitVec.toNat_sub_of_le (by change 1 ≤ d.toNat; omega)

private theorem lmr_less (n : LateMoveReduction.Node) (d : Depth) (depth : n.depth = d.toNat) (pos : 0 < d.toNat)
    (cut pv : Bool) (alpha moveValue : Int) :
    (d-1-BitVec.ofNat 32 (LateMoveReduction.scheduled n cut pv alpha moveValue)).toNat < d.toNat := by
  let r := LateMoveReduction.scheduled n cut pv alpha moveValue
  have bound := (LateMoveReduction.contract_proven n (LateMoveReduction.proposedReduction n cut pv alpha moveValue)
    (depth ▸ pos)).2.2.1
  change r ≤ n.depth / 3 at bound
  rw [depth] at bound
  have below := d.isLt
  have repr : (BitVec.ofNat 32 r).toNat = r := Nat.mod_eq_of_lt (by omega)
  change (d-1-BitVec.ofNat 32 r).toNat < d.toNat
  rw [BitVec.toNat_sub_of_le (by change (BitVec.ofNat 32 r).toNat ≤ (d-1).toNat; rw [sub_one _ pos,repr]; omega),
    sub_one _ pos,repr]
  omega

structure Valid (q : Request) : Prop where
  window : Aspiration.Window q.alpha q.beta
  narrow : q.pv = false → q.beta = q.alpha + 1

/-- An untyped caller can name any request. Issued owner/root calls satisfy Valid. -/
def atRequest (q : Request) (next : Valid q → Plan mc fc q) : Plan mc fc q :=
  if h : (-32767 ≤ q.alpha ∧ q.alpha < q.beta ∧ q.beta ≤ 32767) ∧ (q.pv = false → q.beta = q.alpha+1)
  then next ⟨⟨h.1.1,h.1.2.1,h.1.2.2⟩,h.2⟩ else .stopped

theorem atRequest_valid (q : Request) (next : Valid q → Plan mc fc q) (v : Valid q) :
    atRequest q next = next v := by
  unfold atRequest
  rw [dite_eq_left ⟨⟨v.window.lower,v.window.ordered,v.window.upper⟩,v.narrow⟩]

/-- Every owner child, including the re-search after a scout, executes its native continuation. -/
theorem owner_requests (q : Request) (window : Aspiration.Window q.alpha q.beta)
    (moves : List (GeneratedSquare mc q.board.val))
    (k : (r : Request) → Valid r → Plan mc fc r) :
    SerialCalls.Program.Ready (fun child =>
      let r := SerialPvCalls.moveRequest q.board mc fc q.depth q.cut child
      ∃ v : Valid r, atRequest r (k r) = k r v)
      (SerialOwnerCalls.run Subtype.val q.beta moves (RootReplay.start q.alpha)) := by
  apply (SerialOwnerCalls.run_ready (s := RootReplay.start q.alpha) window moves).weaken
  intro child good
  exact ⟨⟨good.1,good.2⟩,atRequest_valid ..⟩

abbrev Calls (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (limit : Nat) :=
  Path → Selectivity → (r : Request) → Empties r.board → Valid r → r.depth.toNat < limit → Plan mc fc r

def ordinary (ops : Inputs) (path : Path) (q : Request) (e : Option EntryData)
    (moves : List (GeneratedSquare mc q.board.val)) : List (GeneratedSquare mc q.board.val) :=
  let key := ops.ordering path q e
  moves.mergeSort fun x y => decide (key y.val < key x.val ∨
    (key x.val = key y.val ∧ ops.tie path q x.val ≤ ops.tie path q y.val))

theorem ordinary_perm (ops : Inputs) (path : Path) (q : Request) (e : Option EntryData)
    (moves : List (GeneratedSquare mc q.board.val)) : (ordinary ops path q e moves).Perm moves :=
  List.mergeSort_perm _ _

theorem mobile_ranking (ops : Inputs) (path : Path) (q : Request) (entry : Option EntryData)
    (has : Board.hasLegalMoves mc q.board.val = true) :
    ordinary ops path q entry (legalSquares mc q.board.val) ≠ [] := by
  intro empty
  have missing := List.Perm.eq_nil (empty ▸ (ordinary_perm ops path q entry (legalSquares mc q.board.val)).symm)
  exact Bool.noConfusion (has.symm.trans (PvsBoardTree.stuck_of_nil missing))

def probedMove (mc : Bitboard.Moves.X86Config) (q : Request) (entry : Option EntryData) :
    Option (GeneratedSquare mc q.board.val) := do
  let e ← entry
  let sq ← TranspositionTable.Packing.decodeBestMove e.bestMove
  if h : (Board.getMoves mc q.board.val).getLsbD sq.val = true then some ⟨sq,h⟩ else none

/-- A native occupied TT hit cannot lose a decoded move to the generated-square check. -/
theorem probedMove_hit (q : Request) (m : Memory l) (ready : ReadyMemory p mc fc m)
    (hit : m.tt.image.probe q.board = .hit slot e)
    (decoded : TranspositionTable.Packing.decodeBestMove e.bestMove = some sq) :
    ∃ mv, probedMove mc q (some e) = some mv ∧ mv.val = sq := by
  have reads := m.tt.image.probe_reads m.supported q.board
  obtain ⟨mv,_,rfl⟩ := (reads.legalHint ready.1.1.1.hints hit).generated (mc := mc) decoded
  exact ⟨mv,by simp [probedMove,decoded,mv.property],rfl⟩

def pathTo (path : Path) (site : Nat) (sq : Square) : Path := site :: sq.val :: path

def scoutLoop (ops : Inputs) (descend : Calls mc fc limit) (path : Path) (level : Selectivity)
    (q : Request) (state : Empties q.board) (fits : 0 < q.depth.toNat ∧ q.depth.toNat ≤ limit) (sc : q.pv = false) (slot : Fin 2) (ttPv : Bool) (entry : Option EntryData)
    (win : Aspiration.Window q.alpha q.beta) (total : Nat) : (moves : List (GeneratedSquare mc q.board.val)) →
    Nat → (best : Option Value) → (moves = [] → best.isSome = true) → Plan mc fc q
  | [], _, none, filled => by have h := filled rfl; contradiction
  | [], _, some best, _ => .writeAt slot .classified level ttPv none (known sc best)
  | sq :: rest, count, best, _ =>
    let child := (play mc fc q.board sq).val
    let n : LateMoveReduction.Node := ⟨q.depth.toNat,count+1,total,false,false,level != 5⟩
    let reduction := LateMoveReduction.scheduled n q.cut ttPv q.alpha (ops.ordering path q entry sq.val)
    let r := MidgameSites.scout q child (BitVec.ofNat 32 reduction)
    let accept (word : Value) :=
      let v := word.neg
      if ops.stop path q sq.val then .stopped
      else if q.alpha < v.val then .writeAt slot .classified level ttPv (some sq) (known sc v)
      else scoutLoop ops descend path level q state fits sc slot ttPv entry win total rest (count+1)
        (some (best.elim v (fun previous => previous.max v))) (fun _ => rfl)
    call r (MidgameScoreRange.scout_window win) (descend (pathTo path 0 sq.val) level r (placeState mc fc q.board state sq) ⟨MidgameScoreRange.scout_window win,fun _ => by dsimp [r,MidgameSites.scout,MidgameSites.retry]; omega⟩ (by
      have h := lmr_less n q.depth rfl fits.1 q.cut ttPv q.alpha (ops.ordering path q entry sq.val)
      exact Nat.lt_of_lt_of_le h fits.2)) fun word =>
      if 0 < reduction ∧ q.alpha < -word.val then
        let retry := MidgameSites.retry q child
        call retry (MidgameScoreRange.scout_window win) (descend (pathTo path 1 sq.val) level retry (placeState mc fc q.board state sq) ⟨MidgameScoreRange.scout_window win,fun _ => by dsimp [retry,MidgameSites.retry]; omega⟩ (by
          change (q.depth-1).toNat < limit
          rw [sub_one _ fits.1]; omega)) accept
      else accept word

def scoutMoves (ops : Inputs) (descend : Calls mc fc limit) (path : Path) (level : Selectivity)
    (q : Request) (state : Empties q.board) (fits : 0 < q.depth.toNat ∧ q.depth.toNat ≤ limit) (win : Aspiration.Window q.alpha q.beta) (has : Board.hasLegalMoves mc q.board.val = true) (sc : q.pv = false) (slot : Fin 2) (entry : Option EntryData) : Plan mc fc q :=
  let all := legalSquares mc q.board.val
  let ttMv := probedMove mc q entry
  let ttPv := entry.any TranspositionTable.Access.isPv
  let first := if all.length = 1 then all.head? else ttMv
  match first with
  | none => scoutLoop ops descend path level q state fits sc slot ttPv entry win all.length
      (ordinary ops path q entry all) 0 none (fun empty => (mobile_ranking ops path q entry has empty).elim)
  | some sq =>
    let r := MidgameSites.retry q (play mc fc q.board sq).val
    call r (MidgameScoreRange.scout_window win) (descend (pathTo path 2 sq.val) level r (placeState mc fc q.board state sq) ⟨MidgameScoreRange.scout_window win,fun _ => by dsimp [r,MidgameSites.scout,MidgameSites.retry]; omega⟩ (by
      change (q.depth-1).toNat < limit
      rw [sub_one _ fits.1]; omega)) fun word =>
      let v := word.neg
      if ops.stop path q sq.val then .stopped
      else if q.alpha < v.val then .writeAt slot .lower level ttPv (some sq) (known sc v)
      else
        let rest := (ordinary ops path q entry all).erase sq
        scoutLoop ops descend path level q state fits sc slot ttPv entry win all.length rest 1 (some v) (fun _ => rfl)

def afterProbe (ops : Inputs) (descend : Calls mc fc limit) (path : Path) (level : Selectivity)
    (q : Request) (state : Empties q.board) (fits : 0 < q.depth.toNat ∧ q.depth.toNat ≤ limit) (win : Aspiration.Window q.alpha q.beta) (has : Board.hasLegalMoves mc q.board.val = true)
    (slot : Fin 2) (sample : Option Sample) : Plan mc fc q :=
  let entry := sample.map Subtype.val
  let moves := legalSquares mc q.board.val
  let ranked := ordinary ops path q entry moves
  if sc : q.pv = false then
    let next := scoutMoves ops descend path level q state fits win has sc slot entry
    let margin := q.beta + MidgameScoreRange.ttMargin
    let later := match sample with
      | none => next
      | some e =>
        if level != 5 && (e.val.bound == 1 || e.val.bound == 3) && e.val.depth.toNat + 4 ≥ q.depth.toNat &&
          TranspositionTable.selectivityRank e.val.selectivity ≥ level.val then
          if high : margin ≤ e.val.score.toInt then
            known sc ⟨margin,MidgameScoreRange.margin_finite win e.property high⟩
          else next
        else next
    if finite : -32767 < q.beta ∧ q.beta < 32767 then
      let (evaluation,verification,screen) := ops.estimates path q
      match made : MidgameProbCut.plan false q.depth.toNat level.val evaluation verification screen with
      | .skip => later
      | .useStatic threshold => if threshold ≤ evaluation then known sc ⟨q.beta,finite⟩ else later
      | .verify op =>
        let r := MidgameSites.verified q (BitVec.ofNat 32 op.depth) op.beta
        have chosen := (MidgameProbCut.callContract_proven false q.depth.toNat level.val q.beta evaluation verification screen
          (fun _ => 0)).2.1 op (by rw [made]; rfl)
        have low : op.depth < limit := Nat.lt_of_lt_of_le chosen.2.1 fits.2
        have repr : (BitVec.ofNat 32 op.depth).toNat = op.depth := Nat.mod_eq_of_lt (by have := q.depth.isLt; omega)
        have w : Aspiration.Window r.alpha r.beta := by
          change Aspiration.Window (op.beta-1) op.beta
          rw [show op.beta-1 = op.alpha by have h := chosen.2.2.1; omega]
          exact chosen.2.2.2.1
        call r w (descend (3 :: path) 5 r state ⟨w,fun _ => by dsimp [r,MidgameSites.verified]; omega⟩ (repr ▸ low)) fun v => if op.beta ≤ v.val then known sc ⟨q.beta,finite⟩ else later
    else later
  else
      .principalStoreUntil slot level ranked ((ordinary_perm ops path q entry moves).nodup_iff.mpr
        (EndgameOrderMir.generated_layout mc q.board).1) (mobile_ranking ops path q entry has) (fun sq => ops.stop path q sq.val) fun child =>
        let r := SerialPvCalls.moveRequest q.board mc fc q.depth q.cut child
        atRequest r fun valid => descend (pathTo path (if child.pv then 4 else 0) child.move.val) level r (placeState mc fc q.board state child.move) valid (by
          change (q.depth-1).toNat < limit
          rw [sub_one _ fits.1]; omega)

def enhanced (q : Request) (sc : q.pv = false) (level : Selectivity) (slot : Fin 2)
    (ttPv : Bool) (miss : Plan mc fc q) : List (GeneratedSquare mc q.board.val) → Plan mc fc q
  | [] => miss
  | sq :: rest =>
    MidgameValue.lookup (play mc fc q.board sq).val fun sample =>
      match sample with
      | none => enhanced q sc level slot ttPv miss rest
      | some e =>
        let query : PvsEnhancedCutoff.Query := ⟨q.depth,level,false,Writes.word q.alpha⟩
        match found : PvsEnhancedCutoff.candidate query (sq.val,some e.val) with
        | none => enhanced q sc level slot ttPv miss rest
        | some ans => .writeAt slot .lower level ttPv (some sq)
            (known sc ⟨ans.2,MidgameNativeWords.candidate e.val e.property query sq.val found⟩)

def mobile (ops : Inputs) (descend : Calls mc fc limit) (path : Path) (level : Selectivity)
    (q : Request) (state : Empties q.board) (fits : 0 < q.depth.toNat ∧ q.depth.toNat ≤ limit)
    (win : Aspiration.Window q.alpha q.beta) (has : Board.hasLegalMoves mc q.board.val = true) : Plan mc fc q :=
  let moves := legalSquares mc q.board.val
  if moves.any (fun m => EndgameCutoffs.wipesOut fc q.board.val m.val) then .wipeout
  else MidgameValue.probe q.board fun slot sample =>
    let entry := sample.map Subtype.val
    let iir := ExactDepth.iir false false (!q.pv && !q.cut) (level != 5) q.depth.toNat (probedMove mc q entry).isSome
    let d := if iir then q.depth - 1 else q.depth
    have fitD : 0 < d.toNat ∧ d.toNat ≤ limit := by
      dsimp only [d]
      split
      next reduced =>
        have deep : 6 ≤ q.depth.toNat := by
          simp [iir,ExactDepth.iir] at reduced; grind
        rw [sub_one _ fits.1]; omega
      next => exact fits
    let ordinary := Plan.reindex d (afterProbe ops descend path level {q with depth := d} state fitD win has slot sample)
    if sc : q.pv = false then
      let etc := if q.depth.toNat ≥ 5 then
        enhanced q sc level slot (entry.any TranspositionTable.Access.isPv) ordinary moves
        else ordinary
      match sample with
      | none => etc
      | some e =>
        if (TranspositionTable.canCut ops.config.checked e.val (Writes.word q.beta) q.depth level false).toOption == some true then
          known sc e.word
        else if !TranspositionTable.isEndgame e.val && q.depth.toNat ≥ 6 && e.val.depth.toNat ≥ q.depth.toNat &&
          TranspositionTable.selectivityRank e.val.selectivity ≥ level.val then .penalty etc
        else etc
    else ordinary

def promotion (ops : Inputs) (q : Request) (state : Empties q.board) (sc : q.pv = false)
    (size : (Board.emptyCount q.board.val).toNat ≤ 12) (valid : Valid q) : Plan mc fc q :=
  let a := SerialCalls.ScoreWords.encode q.alpha
  let z := SerialCalls.ScoreWords.encode q.beta
  have lower := valid.window.lower
  have upper := valid.window.upper
  have ordered := valid.window.ordered
  have lo : a.raw.toInt = q.alpha := BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
  have hi : z.raw.toInt = q.beta := BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
  let call : Call := ⟨ops.config,q.board,state.val,a,z,state.property, state.property.1 ▸ size,by rw [hi,lo,valid.narrow sc]⟩
  .promotion sc call rfl lo hi

def frame (ops : Inputs) (path : Path) (level : Selectivity) (q : Request)
    (state : Empties q.board) (valid : Valid q) (descend : Path → Selectivity → (r : Request) → Empties r.board → Valid r → rank mc r < rank mc q → Plan mc fc r) : Plan mc fc q :=
  let remaining := (Board.emptyCount q.board.val).toNat
  let generated := fun (pos : 0 < q.depth.toNat) =>
    if has : Board.hasLegalMoves mc q.board.val then
      mobile ops (fun pp ll r s good low => descend pp ll r s good (lower_rank mc q r low)) path level q state ⟨pos,Nat.le_refl _⟩ valid.window has
    else if other : Board.hasLegalMoves mc (pass q.board).val then
      .pvsPass (descend (5 :: path) level (MidgameSites.passed q)
        ⟨state.val,EndgameSearch.emptyState_pass q.board _ _ state.property⟩ ⟨MidgameScoreRange.reflected valid.window,fun scout => by
        have eq := valid.narrow scout; change -q.alpha = -q.beta+1; omega⟩ (by
        simp [rank,MidgameSites.passed,has,other]))
    else .terminal
  if sc : q.pv = false then
    if promoted : remaining ≤ 12 ∧ remaining ≤ q.depth.toNat then promotion ops q state sc promoted.1 valid
    else if small : q.depth.toNat ≤ 3 then MidgameShallow.run (ops.shallow path) q sc valid.window small
    else
      let alphaD := (ScoreRefresh.toDiscDiff (SerialCalls.ScoreWords.encode q.alpha)).raw.toInt
      match hit : Stability.Cutoff.cutoffAtEmptyCount q.board.val.player q.board.val.opponent remaining ⟨alphaD⟩ with
      | some result => known sc ⟨256 * result.value,MidgameNativeWords.stability q _ hit⟩
      | none => generated (by omega)
  else if zero : q.depth.toNat = 0 then .static ((ops.shallow path).raw [] q.board)
  else generated (by omega)

def build (ops : Inputs) (path : Path) (level : Selectivity) (q : Request) (state : Empties q.board) (valid : Valid q) : Plan mc fc q :=
  frame ops path level q state valid (fun nextPath nextLevel r s w _smaller => build ops nextPath nextLevel r s w)
termination_by rank mc q

def plan (ops : Inputs) (q : Request) : Plan mc fc q :=
  atRequest q fun v => build ops [] ops.level q (initial q.board) v

/-- A root slot supplies the child board and the set bit removed by the native context update.
    The default extends the child function to unrelated requests, which the root cannot issue. -/
def rootState (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (root child : Position) : Empties child :=
  let seed := initial root
  match h : SerialChildState.nextState mc fc root 1 seed.val child 0 with
  | none => initial child
  | some s => ⟨s, by
      obtain ⟨t,eq,aligned⟩ := SerialChildState.nextState_matches (SerialChildState.edge_of_nextState h) seed.property
      exact Option.some.inj (eq.symm.trans h) ▸ aligned⟩

theorem rootState_placed (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (root : Position)
    (sq : GeneratedSquare mc root.val) :
    (rootState mc fc root (play mc fc root sq).val).val = (initial root).val.remove sq.val := by
  have h := SerialChildState.nextState_play mc fc root 1 (initial root).val sq
  change SerialChildState.nextState mc fc root 1 (initial root).val (play mc fc root sq).val 0 = _ at h
  dsimp only [rootState]
  split
  · rename_i absent; rw [h] at absent; contradiction
  · rename_i present; exact (Option.some.inj (present.symm.trans h))

def planFromRoot (ops : Inputs) (root : Position) (q : Request) : Plan mc fc q :=
  atRequest q fun v => build ops [] ops.level q (rootState mc fc root q.board) v

theorem planFromRoot_valid (ops : Inputs) (root : Position) (q : Request) (valid : Valid q) :
    planFromRoot (mc := mc) (fc := fc) ops root q =
      build ops [] ops.level q (rootState mc fc root q.board) valid := atRequest_valid ..

def runFromRoot {mc : Bitboard.Moves.X86Config} {fc : Cutoffs.FlipConfig}
    (p : Pool) (ops : Inputs) (root : Position) : MidgameChild.Child l :=
  MidgameChild.run p (planFromRoot (mc := mc) (fc := fc) ops root)

theorem returnsFromRoot (p : Pool) (ops : Inputs) (root : Position) (q : Request) (m : Memory l)
    (win : Aspiration.Window q.alpha q.beta) (good : ReadyMemory p mc fc m) :
    ∃ v after, runFromRoot (mc := mc) (fc := fc) p ops root q m = some (v,after) ∧ ReadyMemory p mc fc after ∧
      SerialCalls.ScoreWords.Finite v ∧ WeakTrial.Range q v :=
  MidgameChild.returns p (planFromRoot ops root) q m win good

def run {mc : Bitboard.Moves.X86Config} {fc : Cutoffs.FlipConfig} (p : Pool) (ops : Inputs) : MidgameChild.Child l := MidgameChild.run p (plan (mc := mc) (fc := fc) ops)

theorem returns (p : Pool) (ops : Inputs) (q : Request) (m : Memory l) (win : Aspiration.Window q.alpha q.beta)
    (good : ReadyMemory p mc fc m) :
    ∃ v after, run (mc := mc) (fc := fc) p ops q m = some (v,after) ∧ ReadyMemory p mc fc after ∧
      SerialCalls.ScoreWords.Finite v ∧ WeakTrial.Range q v :=
  MidgameChild.returns p (plan ops) q m win good

end NeuralReversi.PvsParallel.SerialTable.MidgameNative
