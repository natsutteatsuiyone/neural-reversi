module
public import NeuralReversi.AspirationMir.PublicScore
public import NeuralReversi.PvsParallel.RootSerialMir

@[expose] public section

namespace NeuralReversi.AspirationMir

open Bitboard (Square)
open PvsBoardTree (Position GeneratedSquare play minimax)
open PvsParallel
open RootMoveStorage (Row)

structure SerialTrial (mc : Bitboard.Moves.X86Config) (board : Position) where
  moves : List (GeneratedSquare mc board.val)
  child : PvsMoveLoop.ChildSearch (GeneratedSquare mc board.val)

def SerialTrial.snapshot (trial : SerialTrial mc board) (alpha beta : Word) :=
  RootReplay.serial Subtype.val trial.child beta.toInt trial.moves (RootReplay.start alpha.toInt)

/-- A serial root may change move ordering and child replies between attempts.
Each PV child owes an alpha-beta bound; each scout owes only its null-window bound. -/
theorem publicSerial (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config) (board : Position)
    (initial : List Row) (covers : Root.Covers mc board (initial.map Row.square)) (positive : 0 < initial.length)
    (m : Machine) (entry : m.pc = 0) (initialWindow : Aspiration.Window m.regs.alpha.toInt m.regs.beta.toInt)
    (trial : Nat → Word → Word → SerialTrial mc board)
    (order : ∀ i a b, (trial i a b).moves.Perm (Root.choices mc board (initial.map Row.square) 0))
    (children : ∀ i a b, ∀ move ∈ (trial i a b).moves, ∀ pv lo hi, Aspiration.Window lo hi →
      (pv = false → hi = lo + 1) →
      Aspiration.SoundResult (minimax mc fc (play mc fc board move).val) lo hi ((trial i a b).child move pv lo hi))
    (pv : GeneratedSquare mc board.val → List Square) (depth minSplit nodes : Nat)
    (sort : PublicationMir.SliceSort) (fallback : Unit → PublicationMir.SearchResult) :
    let query : Query := fun i a b => BitVec.ofInt 32 ((trial i a b).snapshot a b).state.best
    ∃ score last, Returns query (fun _ => false) m score last ∧
      let t := trial last.index last.alpha last.beta
      let start := RootReplay.start last.alpha.toInt
      let rows := RootSerialMir.rowsAfter Subtype.val pv t.child last.beta.toInt t.moves start
        (RootMoveStorage.resetFrom initial 0)
      RootSerialMir.run Subtype.val t.child depth minSplit last.beta.toInt t.moves start =
        some (t.snapshot last.alpha last.beta) ∧
      ∃ result, PublicationMir.completed sort rows depth 5 true nodes fallback = some result ∧
        PublicationMir.score result = some score.toInt ∧ score.toInt = minimax mc fc board := by
  intro query
  have positive' : 0 < (initial.map Row.square).length := by simpa using positive
  have attempt (i : Nat) (a b : Word) (window : Aspiration.Window a.toInt b.toInt) :=
    (RootReplay.serial_of_children mc fc board (initial.map Row.square) 0 (trial i a b).moves
      (trial i a b).child (fun move cp lo hi val => val = (trial i a b).child move cp lo hi)
      a.toInt b.toInt covers positive' window (order i a b)
      (PvsMoveLoop.ScanCalls.ofAll (by intros; rfl)) (by
        intro move mem cp lo hi val w z same
        rw [Root.square_value_eq_child move, Root.childValue, Int.neg_neg, same]
        exact children i a b move mem cp lo hi w z)).2
  have repr (i : Nat) (a b : Word) (w : Aspiration.Window a.toInt b.toInt) :
      (query i a b).toInt = ((trial i a b).snapshot a b).state.best := by
    obtain ⟨lo, hi, _⟩ := (attempt i a b w).1
    apply BitVec.toInt_ofInt_eq_self (by decide) <;> change _ < _ at lo hi <;>
      unfold SerialTrial.snapshot <;> omega
  have finite : FiniteQueries query := by
    intro i a b w
    rw [repr i a b w]
    exact ⟨(attempt i a b w).1.lower, (attempt i a b w).1.upper⟩
  obtain ⟨score,last,ran,_,_,w⟩ := completes_in_twelve finite m entry initialWindow
  obtain ⟨⟨observed, low, high⟩, rfl⟩ := returned_inside ran entry
  let t := trial last.index last.alpha last.beta
  have best : (t.snapshot last.alpha last.beta).state.best = last.score.toInt :=
    (repr last.index last.alpha last.beta w).symm.trans (congrArg BitVec.toInt observed)
  have final := attempt last.index last.alpha last.beta w
  have interior : last.alpha.toInt < (t.snapshot last.alpha last.beta).state.best ∧
      (t.snapshot last.alpha last.beta).state.best < last.beta.toInt := best ▸ ⟨low, high⟩
  have exactScore := (final.1.exact interior.1 interior.2).trans (Root.value_zero covers positive')
  have writes := RootSerialMir.rowsAfter_writes Subtype.val pv t.child t.moves (RootReplay.start last.alpha.toInt)
    (RootSuffix.Writes.reset (initial := initial) (idx := 0) (beta := last.beta.toInt))
    (Root.roots_nodup covers) (by
      intro move member
      exact (Root.choices_squares_mem covers move.val).mp
        (List.mem_map_of_mem ((order last.index last.alpha last.beta).mem_iff.mp member)))
  obtain ⟨result, published, raw⟩ := publishSingle covers positive writes interior final sort depth nodes fallback
  refine ⟨_, last, ran, RootSerialMir.run_eq .., result, published, ?_, best ▸ exactScore⟩
  exact raw.trans (congrArg some (exactScore.symm.trans best))

end NeuralReversi.AspirationMir
