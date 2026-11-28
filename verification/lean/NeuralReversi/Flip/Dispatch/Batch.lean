module
public import NeuralReversi.Flip.Avx512.WidePair
public import NeuralReversi.Flip.Dispatch.Properties

@[expose] public section

namespace NeuralReversi.Flip.Dispatch

inductive Backend where
  | avx512
  | avx2
  | neon (sha3 : Bool)
  | scalar
  deriving DecidableEq

def backendFor : TargetConfig → Backend
  | .x86_64 c => if c.avx512cd && c.avx512vl then .avx512 else if c.avx2 then .avx2 else .scalar
  | .aarch64 neon sha3 => if neon then .neon sha3 else .scalar
  | .other => .scalar

def Backend.single (backend : Backend) (sq : Square) (p o : Word) : Word :=
  match backend with
  | .avx512 => Avx512.flip sq p o
  | .avx2 => Avx2.flip sq p o
  | .neon sha3 => Neon.flip sha3 sq p o
  | .scalar => value sq p o

def tableIndex (sq : Square) : Fin 66 := ⟨sq.val, lookup_safe sq⟩

def Backend.pair (backend : Backend) (a b : Square) (p o : Word) : Word × Word :=
  match backend with
  | .avx512 => Avx512.WidePair.narrowPair a b p o
  | .neon sha3 => (Neon.BoardCtx.new p o).flip2 sha3 (tableIndex a) (tableIndex b)
  | .avx2 | .scalar => (backend.single a p o, backend.single b p o)

def Backend.triple (backend : Backend) (a b c : Square) (p o : Word) : Word × Word × Word :=
  match backend with
  | .avx512 => Avx512.WidePair.triple a b c p o
  | .neon sha3 => (Neon.BoardCtx.new p o).flip3 sha3 (tableIndex a) (tableIndex b) (tableIndex c)
  | .avx2 | .scalar => (backend.single a p o, backend.single b p o, backend.single c p o)

def Backend.quad (backend : Backend) (a b c d : Square) (p o : Word) : Word × Word × Word × Word :=
  match backend with
  | .avx512 => Avx512.WidePair.quad a b c d p o
  | .neon sha3 =>
    (Neon.BoardCtx.new p o).flip4 sha3 (tableIndex a) (tableIndex b) (tableIndex c) (tableIndex d)
  | .avx2 | .scalar => (backend.single a p o, backend.single b p o, backend.single c p o, backend.single d p o)

theorem single_eq (backend : Backend) (sq : Square) (p o : Word) (wf : p &&& o = 0) :
    backend.single sq p o = value sq p o := by
  cases backend with
  | avx512 => exact Avx512.flip_eq_scalarValue sq p o
  | avx2 => exact Avx2.publicContract_proven.2 sq p o wf
  | neon sha3 => exact Neon.flip_eq_scalarValue sha3 sq p o wf
  | scalar => rfl

theorem single_matches_flipTarget (target : TargetConfig) (checks : Bool) (sq : Square)
    (p o : Word) (wf : p &&& o = 0) :
    flipTarget target checks sq p o = .ok ((backendFor target).single sq p o) := by
  rw [single_eq _ _ _ _ wf]
  exact flipTarget_eq_value_disjoint target checks sq p o wf

private theorem typedValue (sq : Square) (p o : Word) :
    Avx512.indexedValue (tableIndex sq) p o = value sq p o :=
  Avx512.indexedValue_typed _ sq.isLt _ _

theorem pair_eq (backend : Backend) (a b : Square) (p o : Word) (wf : p &&& o = 0) :
    backend.pair a b p o = (value a p o, value b p o) := by
  cases backend with
  | avx512 => exact Avx512.WidePair.narrowPair_eq a b p o
  | avx2 => simp only [Backend.pair, single_eq _ _ _ _ wf]
  | neon sha3 => simp only [Backend.pair, Neon.flip2_eq _ _ _ _ _ wf, typedValue]
  | scalar => rfl

theorem triple_eq (backend : Backend) (a b c : Square) (p o : Word) (wf : p &&& o = 0) :
    backend.triple a b c p o = (value a p o, value b p o, value c p o) := by
  cases backend with
  | avx512 => exact Avx512.WidePair.triple_eq a b c p o
  | avx2 => simp only [Backend.triple, single_eq _ _ _ _ wf]
  | neon sha3 => simp only [Backend.triple, Neon.flip3_eq _ _ _ _ _ _ wf, typedValue]
  | scalar => rfl

theorem quad_eq (backend : Backend) (a b c d : Square) (p o : Word) (wf : p &&& o = 0) :
    backend.quad a b c d p o = (value a p o, value b p o, value c p o, value d p o) := by
  cases backend with
  | avx512 => exact Avx512.WidePair.quad_eq a b c d p o
  | avx2 => simp only [Backend.quad, single_eq _ _ _ _ wf]
  | neon sha3 => simp only [Backend.quad, Neon.flip4_eq _ _ _ _ _ _ _ wf, typedValue]
  | scalar => rfl

def BatchContract : Prop :=
  ∀ target a b c d p o, p &&& o = 0 →
    let backend := backendFor target
    backend.pair a b p o = (value a p o, value b p o) ∧
      backend.triple a b c p o = (value a p o, value b p o, value c p o) ∧
      backend.quad a b c d p o = (value a p o, value b p o, value c p o, value d p o)

theorem batchContract_proven : BatchContract := by
  intro target a b c d p o wf
  exact ⟨pair_eq _ a b p o wf, triple_eq _ a b c p o wf, quad_eq _ a b c d p o wf⟩

end NeuralReversi.Flip.Dispatch
