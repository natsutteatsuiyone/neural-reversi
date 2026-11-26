import NeuralReversi.PvsParallel.SlotFrameMir.RawProjection

namespace NeuralReversi.PvsParallel.SlotFrameMir.RawProjection

open Threading
open AtomicWord (Write Event)
open SplitCompletion (Action Legal Allowed MaskState Activation initial run step)

theorem action_at (g : RawMask ws) (j : Fin (hits g).length) :
    (actions g).get ⟨j.val,by rw [length_actions];exact j.isLt⟩ = g.actions.get ((hits g).get j) := by
  simp [actions]

def priorCut (g : RawMask ws) (j : Fin (hits g).length) : Cut g (g.before ((hits g).get j)) where
  rank := j.castSucc
  lower := by
    have lt:=ascending g (i:=j.castSucc) (j:=j.succ) (by simp)
    change _ < g.mark.val+((hits g).get j).val+1 at lt
    change _ ≤ g.mark.val+((hits g).get j).val;omega
  later k lt := by
    change j.val<k.val at lt
    have eq : (phys g j.succ).val=(g.before ((hits g).get j)).val+1 := rfl
    by_cases same : j.succ=k
    · rw [←same,eq];omega
    · have ne:=Fin.val_ne_of_ne same
      have beyond := ascending g (i:=j.succ) (j:=k) (by change j.val+1<k.val;change j.val+1≠k.val at ne;omega)
      omega

theorem updates (g : RawMask ws) (j : Fin (hits g).length) :
    let op:=g.actions.get ((hits g).get j)
    let after:=ws.get (phys g j.succ)
    after.event=op.event ∧ after.release=op.release ∧ after.rmw=true ∧
      after.value=op.updateBits (ws.get (phys g j.castSucc)).value := by
  intro op after
  let i:=(hits g).get j
  have before := (priorCut g j).value (by change g.mark.val+i.val ≤ _;have:=i.isLt;omega)
  obtain ⟨ev,rel,rmw,value⟩:=g.rmw i
  exact ⟨ev,rel,rmw,value.trans (congrArg op.updateBits before.symm)⟩

private theorem run_snoc (s : MaskState) (xs : List Action) (a : Action) :
    run s (xs++[a])=step (run s xs) a := by
  induction xs generalizing s with
  | nil => rfl
  | cons x xs ih => exact ih (step s x)

private theorem legal_snoc (s : MaskState) (xs : List Action) (a : Action)
    (older : Legal s xs) (one : Allowed (run s xs) a) : Legal s (xs++[a]) := by
  induction xs generalizing s with
  | nil => exact ⟨one,trivial⟩
  | cons x xs ih => exact ⟨older.1,ih (step s x) older.2 one⟩

private theorem still_open (s : MaskState) (xs : List Action) (openStart : s.accepting=true)
    (noRelease : ∀ a ∈ xs,a.release=false) : (run s xs).accepting=true := by
  induction xs generalizing s with
  | nil => exact openStart
  | cons a xs ih =>
    exact ih (step s a) (by simp [step,noRelease a (by simp),openStart])
      (fun x mem => noRelease x (List.mem_cons_of_mem _ mem))

private theorem book_before (g : RawMask ws) (closed : ClosedAfterFirst g)
    (j : Fin (hits g).length) (op : (g.actions.get ((hits g).get j)).release=false) :
    ∀ x ∈ (actions g).take j.val, x.release=false := by
  intro x member
  obtain ⟨k,get⟩:=List.get_of_mem member
  have len:=k.isLt
  have limit : k.val<j.val := by simp [List.length_take] at len;omega
  let prev : Fin (hits g).length := ⟨k.val,by have:=j.isLt;omega⟩
  have chosen : g.actions.get ((hits g).get prev)=x := by
    rw [←action_at]
    change (actions g)[k.val]'(by rw [length_actions];exact prev.isLt)=x
    simpa only [List.get_eq_getElem,List.getElem_take] using get
  have ord:=List.pairwise_iff_getElem.mp ((List.pairwise_lt_finRange g.actions.length).filter (keeps g))
      prev.val j.val prev.isLt j.isLt limit
  cases rel : x.release with
  | false => rfl
  | true =>
    have forced:=closed ((hits g).get prev) ((hits g).get j) ord (chosen ▸ rel)
    rw [op] at forced;cases forced

private theorem legal_prefix (g : RawMask ws) (closed : ClosedAfterFirst g) (n : Nat) (size : n≤(hits g).length) :
    Legal (initial g.owner) ((actions g).take n) ∧
    (run (initial g.owner) ((actions g).take n)).bits=(ws.get (phys g ⟨n,by omega⟩)).value := by
  induction n with
  | zero => exact ⟨trivial,g.initial.symm⟩
  | succ n ih =>
    obtain ⟨prior,bits⟩:=ih (by omega)
    let j : Fin (hits g).length:=⟨n,by omega⟩
    let i:=(hits g).get j
    have within : n<(actions g).length := by rw [length_actions];omega
    let guarded:=g.actions.get i
    have op : (actions g)[n]=guarded := action_at g j
    have allowed : Allowed (run (initial g.owner) ((actions g).take n)) guarded := by
      cases tag : guarded with
      | book w e =>
        exact still_open _ _ rfl (book_before g closed j (by change guarded.release=false;rw [tag];rfl))
      | finish w e =>
        have selected : keeps g i=true := (List.mem_filter.mp (List.get_mem (hits g) j)).2
        have tag' : g.actions.get i=.finish w e := tag
        simp only [keeps,tag'] at selected
        have before:=(priorCut g j).value (by change g.mark.val+i.val≤_;have:=i.isLt;omega)
        change (run (initial g.owner) ((actions g).take n)).bits.getLsbD w.val=true
        rw [bits];exact (congrArg (fun v : BitVec 64 => v.getLsbD w.val) before).trans selected
    rw [List.take_succ_eq_append_getElem within,op]
    refine ⟨legal_snoc _ _ _ prior allowed,?_⟩
    rw [run_snoc]
    change guarded.updateBits (run (initial g.owner) ((actions g).take n)).bits=_
    rw [bits];exact (updates g j).2.2.2.symm

/-- The projected mask keeps every booking (even if it set an already set bit)
    and every release that removed a set bit. Only redundant reset RMWs vanish. -/
def activation (g : RawMask ws) (closed : ClosedAfterFirst g) : Activation (words g) where
  owner:=g.owner
  start:=index g 0 (by omega)
  actions:=actions g
  endInBounds:=by change 0+_ < _;rw [length_words,length_actions];omega
  legal:=by
    have total:=(legal_prefix g closed (hits g).length (Nat.le_refl _)).1
    simpa only [←length_actions g,List.take_length] using total
  ownerBit:=(congrArg Write.value (word g 0 (by omega))).trans g.initial
  updates j:=by
    let k : Fin (hits g).length := ⟨j.val,Nat.lt_of_lt_of_eq j.isLt (length_actions g)⟩
    have actual:=updates g k
    dsimp only
    simp only [index,Nat.zero_add]
    have x:=word g (k.val+1) (by have:=k.isLt;omega)
    have y:=word g k.val (by have:=k.isLt;omega)
    have a:=action_at g k
    change ((words g).get (index g (k.val+1) (by have:=k.isLt;omega))).event = ((actions g).get j).event ∧
      ((words g).get (index g (k.val+1) (by have:=k.isLt;omega))).release = ((actions g).get j).release ∧
      ((words g).get (index g (k.val+1) (by have:=k.isLt;omega))).rmw=true ∧
      ((words g).get (index g (k.val+1) (by have:=k.isLt;omega))).value =
        ((actions g).get j).updateBits ((words g).get (index g k.val (by have:=k.isLt;omega))).value
    change (actions g).get j=_ at a
    rw [x,y,a]
    exact actual

theorem credit (g : RawMask ws) (closed : ClosedAfterFirst g) (i : Fin g.actions.length)
    (kept : keeps g i=true) :
    ∃ j : Fin (activation g closed).actions.length, (activation g closed).actions.get j=g.actions.get i := by
  obtain ⟨k,eq⟩:=membership g i kept
  exact ⟨⟨k.val,by change _<(actions g).length;rw [length_actions];exact k.isLt⟩,(action_at g k).trans (congrArg g.actions.get eq)⟩

/-- The physical rank of the last retained booking or effective reset. This is the
    owner mark if the generation has no retained actions. -/
def lastEffective (g : RawMask ws) : Fin ws.length := phys g (Fin.last (hits g).length)

/-- Every zero word of this generation is between its final effective update and
    its final physical RMW, inclusive. The interval is empty of zero words if
    the generation never emptied. Concrete reads additionally obey atomic coherence. -/
theorem zero_source_iff (g : RawMask ws) (closed : ClosedAfterFirst g) (s : Fin ws.length) :
    (g.mark.val ≤ s.val ∧ s.val ≤ g.mark.val + g.actions.length ∧ (ws.get s).value = 0) ↔
      (ws.get (lastEffective g)).value = 0 ∧ (lastEffective g).val ≤ s.val ∧
        s.val ≤ g.mark.val + g.actions.length := by
  constructor
  · rintro ⟨base, bound, zero⟩
    let c := cut g s base
    have len : c.rank.val ≤ (hits g).length := by have := c.rank.isLt; omega
    have projected := (legal_prefix g closed c.rank.val len).2
    have empty := projected.trans ((c.value bound).trans zero)
    have last := (SplitCompletion.zeroContract_proven g.owner (actions g) c.rank.val
      (activation g closed).legal (by rwa [length_actions]) empty).1
    have eq : c.rank = Fin.last (hits g).length := Fin.ext (last.trans (length_actions g))
    rw [lastEffective, ←eq]
    exact ⟨(c.value bound).trans zero,c.lower,bound⟩
  · rintro ⟨zero, lower, bound⟩
    have marked : g.mark.val ≤ (lastEffective g).val := by
      let first : Fin ((hits g).length+1) := ⟨0,by omega⟩
      have atMark : phys g first = g.mark := rfl
      rw [←atMark]
      by_cases eq : first = Fin.last (hits g).length
      · exact eq ▸ Nat.le_refl _
      · have ne := Fin.val_ne_of_ne eq
        exact Nat.le_of_lt (ascending g (i:=first) (j:=Fin.last (hits g).length)
          (by change 0 ≠ (hits g).length at ne; change 0 < (hits g).length; omega))
    exact ⟨Nat.le_trans marked lower,bound,g.stays_zero closed _ _ marked lower bound zero⟩

/-- If the generation empties, the first zero has an actual retiring worker
    whose bit was set in the preceding physical word. -/
theorem lastEffective_finish (g : RawMask ws) (zero : (ws.get (lastEffective g)).value = 0) :
    ∃ i : Fin g.actions.length, ∃ w e, g.actions.get i = .finish w e ∧
      g.at i = lastEffective g ∧ (ws.get (g.before i)).value.getLsbD w.val = true := by
  have nonempty : 0 < (hits g).length := by
    apply Classical.byContradiction
    intro no
    have eq : Fin.last (hits g).length = ⟨0,by omega⟩ := Fin.ext (by change (hits g).length = 0; omega)
    rw [lastEffective,eq] at zero
    change (ws.get g.mark).value = 0 at zero
    have bit : (Bitboard.squareMask g.owner).getLsbD g.owner.val = true := by rw [Bitboard.squareMask_bit]; simp
    rw [g.initial.symm.trans zero] at bit
    simp at bit
  let j : Fin (hits g).length := ⟨(hits g).length-1,by omega⟩
  let i := (hits g).get j
  have top : j.succ = Fin.last (hits g).length := Fin.ext (by change (hits g).length-1+1 = (hits g).length; omega)
  have pos : g.at i = lastEffective g := by rw [lastEffective,←top]; rfl
  have keep : keeps g i = true := (List.mem_filter.mp (List.get_mem (hits g) j)).2
  cases op : g.actions.get i with
  | finish w e => exact ⟨i,w,e,op,pos,by simpa only [keeps,op] using keep⟩
  | book w e =>
    have updated := (g.rmw i).2.2.2
    change (ws.get (g.at i)).value = (g.actions.get i).updateBits (ws.get (g.before i)).value at updated
    rw [pos,zero,op] at updated
    have set : ((Action.book w e).updateBits (ws.get (g.before i)).value).getLsbD w.val = true := by
      simp only [Action.updateBits,BitVec.getLsbD_or,Bitboard.squareMask_bit,decide_true,Bool.or_true]
    rw [←updated] at set
    simp at set

/-- Classify all physical zero sources and identify the worker at the first one. -/
def ZeroSourceContract := ∀ (ws : List AtomicWord.Write) (g : RawMask ws), ClosedAfterFirst g →
  (∀ s : Fin ws.length,
    (g.mark.val ≤ s.val ∧ s.val ≤ g.mark.val + g.actions.length ∧ (ws.get s).value = 0) ↔
      (ws.get (lastEffective g)).value = 0 ∧ (lastEffective g).val ≤ s.val ∧
        s.val ≤ g.mark.val + g.actions.length) ∧
  ((ws.get (lastEffective g)).value = 0 → ∃ i : Fin g.actions.length, ∃ w e,
    g.actions.get i = .finish w e ∧ g.at i = lastEffective g ∧
      (ws.get (g.before i)).value.getLsbD w.val = true)

theorem zeroSources : ZeroSourceContract := fun _ g closed =>
  ⟨zero_source_iff g closed,lastEffective_finish g⟩

end NeuralReversi.PvsParallel.SlotFrameMir.RawProjection
