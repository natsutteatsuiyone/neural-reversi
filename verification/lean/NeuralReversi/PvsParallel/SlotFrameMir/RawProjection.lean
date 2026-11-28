module
public import NeuralReversi.PvsParallel.SlotFrameMir.Generation

@[expose] public section

namespace NeuralReversi.PvsParallel.SlotFrameMir.RawProjection

open Threading
open AtomicWord (Event Read Write InReleaseSequence)
open SplitCompletion (Action MaskState Legal Allowed initial run step)

def keeps (g : RawMask ws) (i : Fin g.actions.length) : Bool :=
  match g.actions.get i with
  | .book .. => true
  | .finish w _ => (ws.get (g.before i)).value.getLsbD w.val

def hits (g : RawMask ws) := (List.finRange g.actions.length).filter (keeps g)

def actions (g : RawMask ws) := (hits g).map g.actions.get

def phys (g : RawMask ws) (j : Fin ((hits g).length+1)) : Fin ws.length :=
  match j with
  | ⟨0,_⟩ => g.mark
  | ⟨n+1,h⟩ => g.at ((hits g)[n]'(by omega))

def words (g : RawMask ws) := List.ofFn (fun j => ws.get (phys g j))

theorem length_actions (g : RawMask ws) : (actions g).length=(hits g).length := by simp [actions]
theorem length_words (g : RawMask ws) : (words g).length=(hits g).length+1 := by simp [words]

def index (g : RawMask ws) (n : Nat) (hn : n ≤ (hits g).length) : Fin (words g).length :=
  ⟨n,by rw [length_words];omega⟩

theorem word (g : RawMask ws) (n : Nat) (hn : n ≤ (hits g).length) :
    (words g).get (index g n hn)=ws.get (phys g ⟨n,by omega⟩) := by
  change (List.ofFn (fun j => ws.get (phys g j)))[n]'(by simp;omega)=_
  rw [List.getElem_ofFn]

theorem ascending (g : RawMask ws) {i j : Fin ((hits g).length+1)} (lt : i.val<j.val) :
    (phys g i).val < (phys g j).val := by
  obtain ⟨iv,bi⟩:=i;obtain ⟨jv,bj⟩:=j
  change iv<jv at lt
  cases jv with
  | zero => omega
  | succ b =>
    cases iv with
    | zero => change g.mark.val<g.mark.val+_+1;omega
    | succ a =>
      have pair := List.pairwise_iff_getElem.mp ((List.pairwise_lt_finRange g.actions.length).filter (keeps g))
        a b (by change a<(hits g).length;omega) (by change b<(hits g).length;omega) (by omega)
      change ((hits g)[a]).val < ((hits g)[b]).val at pair
      simp only [phys,RawMask.at];omega

theorem membership (g : RawMask ws) (i : Fin g.actions.length) (kept : keeps g i=true) :
    ∃ j : Fin (hits g).length, (hits g).get j=i :=
  List.get_of_mem (List.mem_filter.mpr ⟨List.mem_finRange i,kept⟩)

private theorem idle_mask (v : BitVec 64) (w : Worker) (off : v.getLsbD w.val=false) :
    v &&& ~~~(Bitboard.squareMask w)=v := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [BitVec.getLsbD_and,BitVec.getLsbD_not,Bitboard.squareMask_bit,hi,decide_true,Bool.true_and]
  by_cases eq : i=w.val
  · subst i;simp only [decide_true,Bool.not_true,Bool.and_false,off]
  · simp [eq]

theorem skipped (g : RawMask ws) (i : Fin g.actions.length) (idle : keeps g i=false) :
    (ws.get (g.at i)).value=(ws.get (g.before i)).value := by
  have val:=(g.rmw i).2.2.2
  change (ws.get (g.at i)).value=(g.actions.get i).updateBits (ws.get (g.before i)).value at val
  cases eq : g.actions.get i with
  | book => simp only [keeps,eq] at idle;cases idle
  | finish w e => rw [eq] at val;exact val.trans (idle_mask _ w (by simpa only [keeps,eq] using idle))

private theorem silent_value (g : RawMask ws) (base : Fin ws.length) (beginning : g.mark.val≤base.val)
    (n : Nat) (range : base.val+n≤g.mark.val+g.actions.length)
    (silent : ∀ i : Fin g.actions.length, base.val < (g.at i).val → (g.at i).val ≤ base.val+n → keeps g i=false) :
    (ws.get ⟨base.val+n,by have:=g.bounded;omega⟩).value=(ws.get base).value := by
  induction n with
  | zero => rfl
  | succ n ih =>
    let i : Fin g.actions.length := ⟨base.val+n-g.mark.val,by omega⟩
    have pos : g.mark.val+i.val=base.val+n := by dsimp [i];omega
    have h:=skipped g i (silent i (by change _<g.mark.val+i.val+1;omega) (by change g.mark.val+i.val+1≤_;omega))
    have prior:=ih (by omega) (fun k lt hi => silent k lt (by omega))
    apply Eq.trans _ prior
    simpa only [RawMask.at,RawMask.before,pos,Nat.add_assoc] using h

structure Cut (g : RawMask ws) (src : Fin ws.length) where
  rank : Fin ((hits g).length+1)
  lower : (phys g rank).val≤src.val
  later : ∀ k, rank.val<k.val → src.val<(phys g k).val

def lastAtOrBefore (g : RawMask ws) (src : Fin ws.length) (first : g.mark.val ≤ src.val)
    (n : Nat) (bound : n ≤ (hits g).length) :
    {i : Fin ((hits g).length+1) // i.val ≤ n ∧ (phys g i).val ≤ src.val ∧
      ∀ j, i.val < j.val → j.val ≤ n → src.val < (phys g j).val} :=
  match n with
  | 0 => ⟨⟨0,by omega⟩,Nat.le_refl _,first,fun _ _ _ => by omega⟩
  | m+1 =>
    let k : Fin ((hits g).length+1) := ⟨m+1,by omega⟩
    if here : (phys g k).val ≤ src.val then
      ⟨k,Nat.le_refl _,here,fun _ _ _ => by dsimp [k] at *; omega⟩
    else
      let ⟨i,within,low,later⟩ := lastAtOrBefore g src first m (by omega)
      ⟨i,by omega,low,fun j above upto => by
        by_cases top : j = k
        · exact top ▸ Nat.lt_of_not_ge here
        · have ne := Fin.val_ne_of_ne top
          exact later j above (by dsimp [k] at ne; omega)⟩

/-- Drop resets of an already clear bit. The physical source projects to the
    last effective modification at or before that source in modification order. -/
def cut (g : RawMask ws) (src : Fin ws.length) (beginning : g.mark.val ≤ src.val) : Cut g src :=
  let ⟨rank,_,low,later⟩ := lastAtOrBefore g src beginning (hits g).length (Nat.le_refl _)
  ⟨rank,low,fun k above => later k above (by have := k.isLt; omega)⟩

theorem Cut.value {g : RawMask ws} (c : Cut g src) (within : src.val≤g.mark.val+g.actions.length) :
    (ws.get (phys g c.rank)).value=(ws.get src).value := by
  let pt:=phys g c.rank
  have base : g.mark.val≤pt.val := by
    dsimp only [pt]
    generalize c.rank=k
    obtain ⟨j,hj⟩:=k
    cases j with
    | zero => exact Nat.le_refl _
    | succ n => change g.mark.val≤g.mark.val+_+1;omega
  have low : pt.val≤src.val := c.lower
  have result:=silent_value g pt base (src.val-pt.val) (by omega) (fun i ahead bound => by
    cases keep : keeps g i with
    | false => rfl
    | true =>
      obtain ⟨j,eq⟩:=membership g i keep
      let k : Fin ((hits g).length+1) := ⟨j.val+1,by have:=j.isLt;omega⟩
      have physical : phys g k=g.at i := by simp [k,phys,←eq]
      have high : c.rank.val<k.val := by
        by_cases lt : k.val<c.rank.val
        · have bad:=ascending g lt;rw [physical] at bad;omega
        · by_cases atRank : k=c.rank
          · have bad : pt=g.at i := by simpa only [pt,atRank] using physical
            rw [←bad] at ahead;omega
          · have diff:=Fin.val_ne_of_ne atRank;omega
      have impossible:=c.later k high;rw [physical] at impossible;omega)
  have k : pt.val+(src.val-pt.val)=src.val := by omega
  simpa [k] using result.symm

end NeuralReversi.PvsParallel.SlotFrameMir.RawProjection
