module
public import NeuralReversi.PvsParallel.NativeFrames.OwnerSites.Admission
public import NeuralReversi.PvsParallel.ContextPathMir

@[expose] public section

namespace NeuralReversi.PvsParallel.NativeFrames.OwnerSites

open FrameReturns.OwnerCalls (ChildReturn)
open PvsBoardTree (Position GeneratedSquare)
open ContextPathMir (At Argument leafState)

/-- The board register and depth are backed by a live Rust search context
    reached from the initialized root, before either the parent or child scores. -/
def Entered (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (root : Position)
    (a : Args) : Prop :=
  ∃ (b : Position) (s : SearchContextState.State), b.val = a.board ∧ At mc fc root b a.depth s

theorem Entered.admissible {a : Args} (path : Entered mc fc root a)
    (window : Aspiration.Window a.alpha.raw.toInt a.beta.raw.toInt)
    (unit : a.pv = false → a.beta.raw.toInt = a.alpha.raw.toInt+1) (word : ScoreRefresh.Score) :
    Admissible (Returns.argsFields a word.raw.toInt) := by
  obtain ⟨b,s,board,atChild⟩ := path
  refine ⟨?_,?_,window,unit⟩
  · change Board.wellFormed a.board; rw [← board]; exact b.property
  · change a.depth.toNat = (Board.emptyCount a.board).toNat
    rw [← board]; exact congrArg BitVec.toNat atChild.full

/-- The recursive call role indexes two independent compiler extractions: the
    borrowed worker edge and the basic block that updates the search context. -/
private theorem move_block (pre : Bool) (index : Nat) (c : ChildReturn) :
    ContextPathMir.childSite false (moveSite pre index c).block = some .move := by
  unfold moveSite
  repeat' split
  all_goals rfl

theorem move_context {f : FrameReturns.Frame} (parent : At mc fc root f.position d s)
    (v : RootOwnerMir.Slice.Visit (GeneratedSquare mc f.position.val)) (pv : Bool) (word : Int)
    (pre : Bool) (index : Nat) :
    let c := FrameReturns.OwnerCalls.child (mc:=mc) (fc:=fc) f d v pv word
    let op := moveSite pre index c
    ∃ next, ContextPathMir.enter mc fc false op.block f.position d s (.move v.move) =
        some ((PvsBoardTree.play mc fc f.position v.move).val,(params c).depth,next) ∧
      (PvsBoardTree.play mc fc f.position v.move).val.val = (params c).board ∧
      Entered mc fc root (params c) := by
  dsimp only
  let b := (PvsBoardTree.play mc fc f.position v.move).val
  let c := FrameReturns.OwnerCalls.child (mc:=mc) (fc:=fc) f d v pv word
  have depth : (params c).depth = d-1 := by
    change BitVec.ofNat 32 (d-1).toNat = d-1
    rw [BitVec.ofNat_toNat,BitVec.setWidth_eq]
  let next := ContextPathMir.place s v.move.val (Flip.Dispatch.resultValue fc v.move.val f.position.val.player f.position.val.opponent)
  have call : ContextPathMir.enter mc fc false (moveSite pre index c).block f.position d s (.move v.move) =
      some (b,(params c).depth,next) := by
    simp only [ContextPathMir.enter,move_block,depth]; rfl
  exact ⟨next,call,rfl,b,next,rfl,parent.entered call⟩

theorem emitted_context {f : FrameReturns.Frame} (parent : At mc fc root f.position d s)
    (visits : List (RootOwnerMir.Slice.Visit (GeneratedSquare mc f.position.val))) :
    ∀ c ∈ FrameReturns.OwnerCalls.emitted fc f d visits, ∀ pre index,
      ∃ arg child next, ContextPathMir.enter mc fc false (moveSite pre index c).block f.position d s arg =
        some (child,(params c).depth,next) ∧ child.val = (params c).board ∧ Entered mc fc root (params c) := by
  intro c found pre index
  obtain ⟨v,_,pv,word,_,rfl⟩ := FrameReturns.OwnerCalls.emitted_iff.mp found
  obtain ⟨next,call,same,path⟩ := move_context parent v pv word pre index
  exact ⟨.move v.move,_,next,call,same,path⟩

theorem pass_context {f : FrameReturns.Frame} (parent : At mc fc root f.position d s)
    (forced : Board.ForcedPass mc f.position.val) (word : Int) :
    let c := FrameReturns.OwnerCalls.passCall f d f.pv word
    ∃ next, ContextPathMir.enter mc fc false Role.passChild.block f.position d s (.forcedPass forced) =
        some (PvsBoardTree.pass f.position,(params c).depth,next) ∧
      (PvsBoardTree.pass f.position).val = (params c).board ∧ Entered mc fc root (params c) := by
  dsimp only
  let c := FrameReturns.OwnerCalls.passCall f d f.pv word
  have depth : (params c).depth = d := by simp [params,c,FrameReturns.OwnerCalls.passCall]
  let next := ContextPathMir.passTurn s
  have call : ContextPathMir.enter mc fc false Role.passChild.block f.position d s (.forcedPass forced) =
      some (PvsBoardTree.pass f.position,(params c).depth,next) := by rw [depth]; rfl
  exact ⟨next,call,rfl,PvsBoardTree.pass f.position,next,rfl,parent.entered call⟩

end NeuralReversi.PvsParallel.NativeFrames.OwnerSites
