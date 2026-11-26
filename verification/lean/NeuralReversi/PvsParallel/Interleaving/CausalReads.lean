import NeuralReversi.PvsParallel.Interleaving.CausalTable
import NeuralReversi.PvsParallel.FrameTableReads.LookupExecution

namespace NeuralReversi.PvsParallel.Interleaving.CausalReads

open NativeTableWrites.Journal
open TranspositionTable
open FrameTableReads (Address snapCode lookupCode lookupValue snapshot)
open FrameTableReads.Operations (Site Supply Code Round probe)
open PvsBoardTree (Position)

structure Reader where
  event : Nat → Nat
  source : Nat → Nat
  generation : Nat → Nat → BitVec 8

def Reader.supply (r : Reader) (x : NativeTableWrites.Journal.Execution c endian rs) : Supply :=
  ⟨fun a i => (x.memory a).readValue (r.event i),r.generation⟩

structure Issued (x : NativeTableWrites.Journal.Execution c endian rs) (r : Reader) (consumer : Nat)
    (site : FrameTableReads.Operations.Site) (code : Code) : Prop where
  steps : ReaderMir.Realizes (x.memory site.address) (code (site.words (r.supply x)))
    (fun i => r.event (site.opening+i)) (fun i => r.source (site.opening+i))
  localOrder : (x.memory site.address).sb (r.event (site.opening+0)) consumer
  noWrap : (code (site.words (r.supply x))).length = 6 → r.source (site.opening+5) < r.source (site.opening+0) + SeqLock.modulus

private theorem snapshot_hit {x : NativeTableWrites.Journal.Execution c endian rs} (init : ∀ a, Even (x.memory a))
    (issued : Issued x r consumer site (snapCode endian)) (b : Position)
    (hit : snapshot endian (site.words (r.supply x)) = some (Cutoffs.key b,e)) (full : e.bound ≠ 0) :
    CausalTable.lookup x init consumer b e := by
  obtain ⟨s,ran,eq⟩ := Option.map_eq_some_iff.mp hit
  obtain ⟨keys,data⟩ := Prod.mk.inj eq
  have occupied : Snapshot.empty endian s.metadata = false := by
    unfold Snapshot.empty
    rw [data,beq_eq_false_iff_ne]
    exact full
  have size : (snapCode endian (site.words (r.supply x))).length = 6 := by
    unfold snapCode ReaderMir.run at *
    repeat' split at ran <;> try contradiction
    all_goals simp_all [Snapshot.empty]
    subst s
    apply full
    rw [←data]
    assumption
  have visible := Table.seen_snapshot x site.address (init _) (fun i => r.event (site.opening+i))
    (fun i => r.source (site.opening+i)) issued.steps ran occupied (issued.noWrap size)
  have point := visible.later ((x.memory _).sb_hb issued.localOrder)
  refine ⟨full,site.address,?_⟩
  have words : (⟨s.player,s.opponent,s.metadata⟩ : SeqLock.Payload) = SlotJournal.packed endian b e := by
    obtain ⟨p,o⟩ := Prod.mk.inj keys
    simp [SlotJournal.packed,←data,Packing.pack_unpack,p,o]
  exact words ▸ point

/-- The emitted front/back probes share one register supply. Generation,
    busy slots and failed retries influence the chosen slot in the MIR program. -/
theorem probe_hit (x : NativeTableWrites.Journal.Execution c endian rs) (init : ∀ a, Even (x.memory a))
    (r : Reader) (consumer allocation : Nat) (clusters key : SeqLock.Word) (b : Position) (start : Nat) :
    let s := r.supply x
    let out := probe s allocation clusters key endian (Cutoffs.key b) start
    (∀ round ∈ out.rounds, Issued x r consumer round.front (snapCode endian) ∧
      ∀ back ∈ round.back, Issued x r consumer back (snapCode endian)) →
    out.result = .hit slot e → CausalTable.lookup x init consumer b e := by
  intro s out sites result
  have traced := (FrameTableReads.Operations.probe_result s allocation clusters key endian (Cutoffs.key b) start).trans result
  change ProbeRetry.probe (out.rounds.map (fun a => ProbeRetry.observe (Cutoffs.key b) (a.read s endian))) = _ at traced
  have result' : ProbeRetry.probe ((out.rounds.map (fun t => t.read s endian)).map (ProbeRetry.observe (Cutoffs.key b))) = .hit slot e := by
    simpa only [List.map_map,Function.comp_def] using traced
  obtain ⟨raw,member,found,occupied⟩ := ProbeRetry.contract_proven.2 _ _ _ _ result'
  obtain ⟨round,appears,rfl⟩ := List.mem_map.mp (List.mem_of_mem_take member)
  obtain ⟨front,back⟩ := sites round appears
  change (if slot = 0 then snapshot endian (round.front.words s) else round.back.bind (fun site => snapshot endian (site.words s))) = _ at found
  split at found
  · exact snapshot_hit init front b found occupied
  · obtain ⟨site,used,actual⟩ := Option.bind_eq_some_iff.mp found
    exact snapshot_hit init (back site used) b actual occupied

private theorem lookup_hit {x : NativeTableWrites.Journal.Execution c endian rs} (init : ∀ a, Even (x.memory a))
    (issued : Issued x r consumer site (lookupCode endian b))
    (hit : lookupValue endian b (site.words (r.supply x)) = some e) : CausalTable.lookup x init consumer b e := by
  have size : (lookupCode endian b (site.words (r.supply x))).length = 6 := by
    unfold lookupCode lookupValue ReaderMir.lookupRun at *
    repeat' split at hit <;> first | contradiction | skip
    simp_all
  have ⟨seen,occupied⟩ := Table.seen_lookup x site.address (init _) b (fun i => r.event (site.opening+i))
    (fun i => r.source (site.opening+i)) issued.steps hit (issued.noWrap size)
  exact ⟨occupied,site.address,seen.later ((x.memory _).sb_hb issued.localOrder)⟩

theorem cluster_hit (x : NativeTableWrites.Journal.Execution c endian rs) (init : ∀ a, Even (x.memory a))
    (r : Reader) (consumer allocation : Nat) (clusters key : SeqLock.Word) (b : Position) (start : Nat) :
    let s := r.supply x
    let out := FrameTableReads.Operations.lookup s allocation clusters key endian b start
    let pair := out.pair s endian b
    Issued x r consumer out.front (lookupCode endian b) →
    (∀ site ∈ out.back, Issued x r consumer site (lookupCode endian b)) →
    pair.1.or pair.2 = some e → CausalTable.lookup x init consumer b e := by
  intro s out pair front back hit
  change (lookupValue endian b (out.front.words s)).or (out.back.bind fun site => lookupValue endian b (site.words s)) = _ at hit
  cases h : lookupValue endian b (out.front.words s) with
  | some d =>
    rw [h] at hit
    cases hit
    exact lookup_hit init front h
  | none =>
    rw [h,Option.none_or] at hit
    obtain ⟨site,used,actual⟩ := Option.bind_eq_some_iff.mp hit
    exact lookup_hit init (back site used) actual

end NeuralReversi.PvsParallel.Interleaving.CausalReads
