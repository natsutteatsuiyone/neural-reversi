import NeuralReversi.PvsParallel.Interleaving.EventOrder
import NeuralReversi.PvsParallel.Interleaving.WriterSteps

namespace NeuralReversi.PvsParallel.Interleaving.TableEventOrder

open TranspositionTable
open SeqLock (Location Memory)
open WriterTrace (Slot)
open NativeTableWrites.Journal (Site Record mods)

/-- Only emitted, positive modification ranks need a position in the physical
    prefix. Initial writes and absent ranks can have IDs outside that prefix. -/
def Emitted {m : Memory Nat} (slot : Slot m) (n : Nat) :=
  ∀ loc i, 0 < i → i ≤ slot.modificationCount loc → m.write loc i < n

def writes {m : Memory Nat} (slot : Slot m) (bounded : Emitted slot n) (loc : Location) : List (Fin n) :=
  List.ofFn fun (i : Fin (slot.modificationCount loc)) =>
    ⟨m.write loc (i.val+1),bounded loc _ (Nat.zero_lt_succ _) i.isLt⟩

/-- One finite numbering preserves happens-before and reproduces all physical
    TT modification orders by projection. Reads-from is never an ordering edge
    solely because the source and load touched the same atomic object. -/
def Contract := ∀ (n : Nat) (Addr : Type) (hb : Fin n → Fin n → Prop) [DecidableRel hb]
  (mem : Addr → Memory Nat) (slots : ∀ a, Slot (mem a)) (bounded : ∀ a, Emitted (slots a) n),
  (∀ a, ¬ hb a a) → (∀ {a b c}, hb a b → hb b c → hb a c) →
  (∀ a (u v : Fin n), (mem a).hb u v → hb u v) →
  (EventOrder.events hb).Perm (List.finRange n) ∧
  (∀ u v, hb u v → EventOrder.position hb u < EventOrder.position hb v) ∧
  ∀ a loc (select : Fin n → Bool), (∀ w, select w = true ↔ w ∈ writes (slots a) (bounded a) loc) →
    (EventOrder.events hb).filter select = writes (slots a) (bounded a) loc

theorem projects : Contract := by
  intro n Addr hb _ mem slots bounded irrefl trans includes
  have ordered := fun {u v} (edge : hb u v) => EventOrder.extendsHB hb irrefl trans edge
  refine ⟨EventOrder.permutation hb,fun _ _ h => ordered h,?_⟩
  intro a loc select complete
  apply EventOrder.projection hb irrefl trans select _ complete
  apply List.pairwise_iff_getElem.mpr
  intro i j _ hj lt
  have valid : j < (slots a).modificationCount loc := by simpa [writes] using hj
  simp only [writes,List.getElem_ofFn]
  exact includes a _ _ ((slots a).modificationsHB loc (by omega) (by omega) (by omega))

/-- The actual writer replay bounds every modifying instruction, including a
    final CAS whose payload or release is absent. -/
theorem journalEmitted (x : NativeTableWrites.Journal.Execution c endian rs) (h : WriterSteps.Facts c es rs)
    (init : NativeTableWrites.Journal.Even (x.memory a)) : Emitted (x.slot a init) es.length := by
  intro loc i positive inside
  have count : (x.slot a init).modificationCount loc = (mods c rs a loc).length := by
    cases loc <;> apply List.length_map
  rw [count] at inside
  obtain ⟨j,rfl⟩ := Nat.exists_eq_succ_of_ne_zero (by omega : i ≠ 0)
  have hit := List.getElem?_eq_getElem (l := mods c rs a loc) (i := j) (by omega)
  obtain ⟨stamp,_⟩ := x.modified _ _ _ _ hit
  rw [stamp]
  exact h.bound (NativeTableWrites.Journal.mem_mods.mp (List.mem_of_getElem? hit)).1

/-- A single event permutation recovers the list of actual writer stamps in
    each slot. The selector uses the emitted PCs of this address and field. -/
theorem journalProjection (x : NativeTableWrites.Journal.Execution c endian rs) (h : WriterSteps.Facts c es rs)
    (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a))
    (hb : Fin es.length → Fin es.length → Prop) [DecidableRel hb]
    (irrefl : ∀ w, ¬ hb w w) (trans : ∀ {u v w}, hb u v → hb v w → hb u w)
    (includes : ∀ a (u v : Fin es.length), (x.memory a).hb u v → hb u v) (a loc) :
    let stamps := (mods c rs a loc).map (Site.event c)
    ((EventOrder.events hb).filter fun (w : Fin es.length) => stamps.contains w.val).map Fin.val = stamps := by
  let slot := x.slot a (init a)
  let bound := journalEmitted x h (init a)
  have count : slot.modificationCount loc = (mods c rs a loc).length := by
    cases loc <;> apply List.length_map
  have eq : (writes slot bound loc).map Fin.val = (mods c rs a loc).map (Site.event c) := by
    apply List.ext_getElem
    · simp [writes,count]
    · intro i hi _
      have inside : i < (mods c rs a loc).length := by simpa [writes,count] using hi
      simpa only [writes,List.getElem_map,List.getElem_ofFn] using
        (x.modified a loc i _ (List.getElem?_eq_getElem inside)).1
  let stamps := (mods c rs a loc).map (Site.event c)
  have selected (w : Fin es.length) : stamps.contains w.val = true ↔ w ∈ writes slot bound loc := by
    change ((mods c rs a loc).map (Site.event c)).contains w.val = true ↔ _
    rw [←eq,List.contains_iff_mem,List.mem_map]
    exact ⟨fun ⟨v,member,same⟩ => Fin.ext same ▸ member,fun member => ⟨w,member,rfl⟩⟩
  exact (congrArg (List.map Fin.val) ((projects _ _ hb x.memory (fun b => x.slot b (init b))
    (fun b => journalEmitted x h (init b)) irrefl trans includes).2.2 a loc (fun w => stamps.contains w.val)
    selected)).trans eq

end NeuralReversi.PvsParallel.Interleaving.TableEventOrder
