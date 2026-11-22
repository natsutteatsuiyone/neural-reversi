import NeuralReversi.Eval.InputLayer.Arithmetic

namespace NeuralReversi.Eval.InputLayer

inductive X86Layout where
  | avx2
  | avx512

def tileWords : X86Layout → Nat
  | .avx2 => 4
  | .avx512 => 8

def tileOrder : X86Layout → Fin 8 → Nat
  | .avx2 => fun i => #[0, 1, 4, 5, 2, 3, 6, 7][i]
  | .avx512 => fun i => #[0, 2, 4, 6, 1, 3, 5, 7][i]

def loadedColumn (layout : X86Layout) (i : Nat) : Nat :=
  let tile := tileWords layout
  let block := 8 * tile
  let tileIndex : Fin 8 := ⟨(i / tile) % 8, Nat.mod_lt _ (by decide)⟩
  (i / block) * block + tileOrder layout tileIndex * tile + i % tile

/-- PACKUSWB takes eight words from each of its two operands in each 128-bit lane. -/
def packedSource (layout : X86Layout) (pos : Nat) : Nat :=
  let width := 8 * tileWords layout
  let lane := pos % width
  pos - lane + (lane / 16) * 8 + ((lane / 8) % 2) * (width / 2) + lane % 8

def restores (layout : X86Layout) (i : Fin 128) : Prop :=
  let src := packedSource layout i.val
  src < 128 ∧ loadedColumn layout src = i.val ∧
    loadedColumn layout (128 + src) = 128 + i.val

instance (layout : X86Layout) (i : Fin 128) : Decidable (restores layout i) := by
  unfold restores
  infer_instance

def PackOrderContract : Prop :=
  (∀ i, restores .avx2 i) ∧ (∀ i, restores .avx512 i)

theorem packOrderContract_proven : PackOrderContract := by
  unfold PackOrderContract
  decide

def x86BaseOutput (layout : X86Layout) (natural : Nat → Int) (i : Nat) : Int :=
  let col := packedSource layout i
  x86Base (natural (loadedColumn layout col)) (natural (loadedColumn layout (128 + col)))

def x86PhaseOutput (layout : X86Layout) (natural : Nat → Int) (i : Nat) : Int :=
  x86Phase (natural (loadedColumn layout (packedSource layout i)))

def NaturalOutputsContract : Prop :=
  ∀ layout (natural : Nat → Int) (i : Fin 128),
    -32768 ≤ natural i.val → -32768 ≤ natural (128 + i.val) →
    x86BaseOutput layout natural i.val = scalarBase (natural i.val) (natural (128 + i.val)) ∧
    x86PhaseOutput layout natural i.val = scalarPhase (natural i.val) ∧
    neonBase (natural i.val) (natural (128 + i.val)) =
      scalarBase (natural i.val) (natural (128 + i.val)) ∧
    neonPhase (natural i.val) = scalarPhase (natural i.val)

theorem naturalOutputsContract_proven : NaturalOutputsContract := by
  intro layout natural i lo hi
  have order : restores layout i := by
    cases layout
    · exact packOrderContract_proven.1 i
    · exact packOrderContract_proven.2 i
  obtain ⟨_, low, high⟩ := order
  have a := activationContract_proven.1 (natural i.val) (natural (128 + i.val)) hi
  have b := activationContract_proven.2 (natural i.val) lo
  unfold x86BaseOutput x86PhaseOutput
  dsimp only
  rw [low, high]
  exact ⟨a.1, b.1, a.2.1, b.2⟩

end NeuralReversi.Eval.InputLayer
