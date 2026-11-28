module
public import NeuralReversi.Stability.EdgeProjection

@[expose] public section

namespace NeuralReversi.Stability.Semantics
def LineCaptured (line : Fin 8 → Option Color) (c : Color) (a x : Fin 8) : Prop :=
 ∃ d : Direction, d.delta.1 = 0 ∧ ∃ n : Nat, 1 ≤ n ∧ n ≤ 6 ∧
 (∃ endpoint : Fin 8, (endpoint.val : Int) = a.val + (n+1 : Nat) * d.delta.2 ∧ line endpoint = some c) ∧
 (∀ k : Nat, 1 ≤ k → k ≤ n → ∃ y : Fin 8,
   (y.val : Int) = a.val + (k : Int) * d.delta.2 ∧ line y = some (Opposite c)) ∧
 (∃ k : Nat, 1 ≤ k ∧ k ≤ n ∧ (x.val : Int) = a.val + (k : Int) * d.delta.2)

theorem row_captured_iff (b : Board) (c : Color) (r a x : Fin 8)
 (hr : r.val = 0 ∨ r.val = 7) :
 Captured b c (r,a) (r,x) ↔ LineCaptured (fun i => b (r,i)) c a x := by
 constructor
 · intro h
   obtain ⟨d,n,hn,hn6,⟨ep,he,hep⟩,run,k,hk,hkn,hx⟩ := h
   have hd := row_edge_capture_parallel hr hk hkn hx he
   refine ⟨d,hd,n,hn,hn6,?_,?_,k,hk,hkn,hx.2⟩
   · have er : ep.1 = r := by
       apply Fin.ext
       have h := he.1
       simp [hd] at h
       omega
     refine ⟨ep.2,he.2,?_⟩
     simpa [← er] using hep
   · intro j hj hjn
     obtain ⟨y,hy,hcolor⟩ := run j hj hjn
     have yr : y.1 = r := by
       apply Fin.ext
       have h := hy.1
       simp [hd] at h
       omega
     exact ⟨y.2,hy.2,by simpa [← yr] using hcolor⟩
 · rintro ⟨d,hd,n,hn,hn6,⟨ep,he,hep⟩,run,k,hk,hkn,hx⟩
   refine ⟨d,n,hn,hn6,⟨(r,ep),?_,hep⟩,?_,k,hk,hkn,?_⟩
   · exact ⟨by simp [hd],he⟩
   · intro j hj hjn
     obtain ⟨y,hy,hcolor⟩ := run j hj hjn
     exact ⟨(r,y),⟨by simp [hd],hy⟩,hcolor⟩
   · exact ⟨by simp [hd],hx⟩
end NeuralReversi.Stability.Semantics
namespace NeuralReversi.Stability.Semantics
def Direction.transpose : Direction → Direction
 | .n => .w | .ne => .sw | .e => .s | .se => .se
 | .s => .e | .sw => .ne | .w => .n | .nw => .nw

theorem transpose_delta (d : Direction) :
 d.transpose.delta = (d.delta.2,d.delta.1) := by cases d <;> rfl

theorem captured_transpose (b : Board) (c : Color) (a x : Square) :
 Captured b c a x → Captured (fun q => b (q.2,q.1)) c (a.2,a.1) (x.2,x.1) := by
 rintro ⟨d,n,hn,hn6,⟨ep,he,hep⟩,run,k,hk,hkn,hx⟩
 refine ⟨d.transpose,n,hn,hn6,⟨(ep.2,ep.1),?_,hep⟩,?_,k,hk,hkn,?_⟩
 · simp only [Along,transpose_delta] at *
   exact ⟨he.2,he.1⟩
 · intro j hj hjn
   obtain ⟨y,hy,hcolor⟩ := run j hj hjn
   refine ⟨(y.2,y.1),?_,hcolor⟩
   simp only [Along,transpose_delta] at *
   exact ⟨hy.2,hy.1⟩
 · simp only [Along,transpose_delta] at *
   exact ⟨hx.2,hx.1⟩

theorem col_captured_iff (b : Board) (c : Color) (r a x : Fin 8)
 (hr : r.val = 0 ∨ r.val = 7) :
 Captured b c (a,r) (x,r) ↔ LineCaptured (fun i => b (i,r)) c a x := by
 constructor
 · intro h
   exact (row_captured_iff (fun q => b (q.2,q.1)) c r a x hr).mp
     (captured_transpose b c (a,r) (x,r) h)
 · intro h
   have ht := (row_captured_iff (fun q => b (q.2,q.1)) c r a x hr).mpr h
   exact captured_transpose (fun q => b (q.2,q.1)) c (r,a) (r,x) ht
end NeuralReversi.Stability.Semantics
namespace NeuralReversi.Stability.Semantics
def FiniteLineCaptured (line : Fin 8 → Option Color) (c : Color) (a x : Fin 8) : Prop :=
 ∃ positive : Bool, ∃ n : Fin 7, 1 ≤ n.val ∧
 (∃ endpoint : Fin 8, (endpoint.val : Int) = a.val + (n.val+1 : Nat) * (if positive then 1 else -1) ∧ line endpoint = some c) ∧
 (∀ k : Fin 8, 1 ≤ k.val → k.val ≤ n.val → ∃ y : Fin 8,
   (y.val : Int) = a.val + (k.val : Int) * (if positive then 1 else -1) ∧ line y = some (Opposite c)) ∧
 (∃ k : Fin 8, 1 ≤ k.val ∧ k.val ≤ n.val ∧ (x.val : Int) = a.val + (k.val : Int) * (if positive then 1 else -1))

theorem line_finite_iff (line : Fin 8 → Option Color) (c : Color) (a x : Fin 8) :
 LineCaptured line c a x ↔ FiniteLineCaptured line c a x := by
 constructor
 · rintro ⟨d,hd,n,hn,hn6,he,run,k,hk,hkn,hx⟩
   have aux (positive : Bool) (hp : d.delta.2 = (if positive then 1 else -1)) : FiniteLineCaptured line c a x := by
     refine ⟨positive,⟨n,by omega⟩,hn,?_,?_,⟨k,by omega⟩,hk,hkn,?_⟩
     · simpa [hp] using he
     · intro j hj hjn
       simpa [hp] using run j.val hj hjn
     · simpa [hp] using hx
   cases d <;> simp [Direction.delta] at hd
   · exact aux true rfl
   · exact aux false rfl
 · rintro ⟨positive,n,hn,he,run,k,hk,hkn,hx⟩
   let d : Direction := if positive then .e else .w
   have hd : d.delta.1 = 0 := by cases positive <;> rfl
   have hp : d.delta.2 = (if positive then 1 else -1) := by cases positive <;> rfl
   refine ⟨d,hd,n.val,hn,by omega,?_,?_,k.val,hk,hkn,?_⟩
   · simpa [hp] using he
   · intro j hj hjn
     simpa [hp] using run ⟨j,by omega⟩ hj hjn
   · simpa [hp] using hx
end NeuralReversi.Stability.Semantics
