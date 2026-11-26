import NeuralReversi.PvsParallel.Interleaving.Entry
import NeuralReversi.PvsParallel.Interleaving.JournalLocalCache
import NeuralReversi.PvsParallel.Interleaving.CutClosure

namespace NeuralReversi.PvsParallel.Interleaving.MultiSessionCarry

open TranspositionTable
open SeqLock (Word Payload Location)
open FrameTableReads (Address)
open NativeTableWrites.Journal (Calls Record Even Execution mods)
open EndgameSearch.CacheHistory (Pool Timeline Good)

/-- Concrete provenance for every slot payload in a chained multi-session TT:
    either the empty zero slot, a validated warm serial TT image, or the final
    released image of a prior parallel session (midgame, selective, interrupted,
    or exact) that drained all active seqlock writers before handoff. -/
inductive SlotOrigin (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (endian : Bool) :
    Payload → Prop where
  | empty : SlotOrigin mc fc endian ⟨0,0,0⟩
  | fromSerial {l : PvsParallel.SerialTable.Layout} (m : PvsParallel.SerialTable.Memory l)
      (i : l.Index) (ready : AspirationMir.SerialTable.Repeated.ReadyMemory pool mc fc m)
      (sameEndian : l.bigEndian = endian) :
      SlotOrigin mc fc endian (PvsParallel.SerialTable.payload (m.tt.image.get i))
  | fromParallel (prev : JournalResident.Session mc fc endian)
      (prevOrigins : ∀ a, SlotOrigin mc fc endian (prev.initial a))
      (a : Address) (drained : prev.drained) :
      SlotOrigin mc fc endian (prev.finalImage a)

theorem SlotOrigin.toResident (o : SlotOrigin mc fc endian p) : Entry.Resident mc fc endian p := by
  induction o with
  | empty => exact .zero
  | fromSerial m i ready sameEndian =>
    subst sameEndian
    exact .warm m i ready
  | fromParallel prev _ a drained ih => exact .parallel prev ih a drained

theorem SlotOrigin.valid (o : SlotOrigin mc fc endian p) : JournalResident.Valid mc fc endian p :=
  o.toResident.valid

theorem SlotOrigin.chain {first last : JournalResident.Session mc fc endian}
    (steps : JournalResident.Chain first last)
    (seed : ∀ a, SlotOrigin mc fc endian (first.initial a)) :
    ∀ a, SlotOrigin mc fc endian (last.initial a) := by
  induction steps with
  | start => exact seed
  | @next prev sess _ link ih =>
    intro a
    rw [JournalResident.Session.initial,link.words sess.init a]
    exact .fromParallel prev ih a link.drained

/-- Provenance of any TT entry adopted by a worker during `current` after a finite
    chain of prior sessions `first ->* prev` handed off via `link`:
    - `inherited`: came from epoch 0 (the resident image from `first ->* prev`,
      which may have been written by any earlier midgame, selective, interrupted,
      or exact search), OR
    - `fromCurrent`: came from a concrete committed seqlock publication `.finish`
      of invocation `k` in `current` (including writes by other workers). -/
inductive AdoptedOrigin (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (endian : Bool)
    (current : JournalResident.Session mc fc endian) (a : Address) : Payload → Prop where
  | inherited (origin : SlotOrigin mc fc endian (current.initial a)) :
      AdoptedOrigin mc fc endian current a (current.initial a)
  | fromCurrent (n k : Nat)
      (published : (mods current.calls current.records a .sequence)[n]? = some ⟨k,.finish⟩) :
      AdoptedOrigin mc fc endian current a ((current.calls k).call endian).payload

/-- Multi-session TT handoff provenance theorem:
    Starting from any valid initial image (`empty`, serial warm TT, or prior chain)
    and advancing through ANY finite chain of drained parallel sessions `first ->* last`
    into `current`:
    1. Every slot starts unlocked (`Even`) with `Entry.Resident` (hence `Table.Safe ∧ FiniteTable.Words ∧ SeedTable.Words`),
    2. Every TT hit adopted by ANY worker during `current` (via `ReaderMir.lookupRun`
       under local no-wrap) traces either to a committed write in `current` (by any worker)
       or to the resident image carried from the prior session chain, AND
    3. Every adopted entry satisfies `Table.Safe mc fc endian` (so exact values equal `V(B)`
       and bounds are sound with respect to `V(B)`). -/
def ChainResidentContract : Prop :=
  ∀ (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (endian : Bool)
    (first last current : JournalResident.Session mc fc endian),
    (∀ a, SlotOrigin mc fc endian (first.initial a)) →
    JournalResident.Chain first last →
    JournalResident.Handoff last current.memory →
    (∀ a, Even (current.memory.memory a) ∧
          Entry.Resident mc fc endian (current.initial a) ∧
          JournalResident.Valid mc fc endian (current.initial a)) ∧
    (∀ (a : Address) (b : PvsBoardTree.Position) (ev source : Nat → Nat) (e : EntryData),
      ReaderMir.Realizes (current.memory.memory a)
        (ReaderMir.lookupRun endian b.val.player b.val.opponent
          ((current.memory.memory a).readValue ∘ ev)).1 ev source →
      (ReaderMir.lookupRun endian b.val.player b.val.opponent
        ((current.memory.memory a).readValue ∘ ev)).2 = some e →
      source 5 < source 0 + SeqLock.modulus →
      let p := (current.memory.slot a (current.init a)).payloadAt (source 0 / 2)
      AdoptedOrigin mc fc endian current a p ∧
      Table.Safe mc fc endian p ∧
      Table.EntrySafe mc fc b e)

theorem chain_resident : ChainResidentContract := by
  intro mc fc endian first last current seed steps handoff
  have lastOrigin : ∀ a, SlotOrigin mc fc endian (last.initial a) := SlotOrigin.chain steps seed
  have curOrigin : ∀ a, SlotOrigin mc fc endian (current.initial a) := by
    intro a
    rw [JournalResident.Session.initial,handoff.words current.init a]
    exact .fromParallel last lastOrigin a handoff.drained
  have curRes : ∀ a, Entry.Resident mc fc endian (current.initial a) :=
    fun a => (curOrigin a).toResident
  have curValid : ∀ a, JournalResident.Valid mc fc endian (current.initial a) :=
    fun a => (curOrigin a).valid
  refine ⟨fun a => ⟨handoff.even a,curRes a,curValid a⟩,?_⟩
  intro a b ev source e realized hit noWrap
  have seen := Table.seen_lookup_sources current.memory a (current.init a) b ev source realized hit noWrap
  let k := source 0 / 2
  have atEpoch : AdoptedOrigin mc fc endian current a ((current.memory.slot a (current.init a)).payloadAt k) ∧
      JournalResident.Valid mc fc endian ((current.memory.slot a (current.init a)).payloadAt k) := by
    cases hk : k with
    | zero =>
      exact ⟨.inherited (curOrigin a),curValid a⟩
    | succ m =>
      let slot := current.memory.slot a (current.init a)
      obtain ⟨_,_,_,_,even0⟩ := seen.2.indices
      have rf0 := seen.2.first
      have bound := current.memory.source a .sequence (source 0) (ev 0) rf0
      have histLen : slot.history.length = (mods current.calls current.records a .sequence).length := by
        simp [slot,NativeTableWrites.Journal.Execution.slot,NativeTableWrites.Journal.Execution.toSequence,
          NativeTableWrites.Journal.sequence]
      have leHist : 2 * (m + 1) ≤ slot.history.length := by rw [histLen]; omega
      obtain ⟨call,pubAct,payloadEq⟩ := slot.publishedPayload (m + 1) (Nat.succ_pos m) leHist
      have pubMap : ((mods current.calls current.records a .sequence).map
          (NativeTableWrites.Journal.Site.action current.calls endian))[2 * (m + 1) - 1]? =
          some (.publish call) := by
        simpa [slot,NativeTableWrites.Journal.Execution.slot,NativeTableWrites.Journal.Execution.toSequence,
          NativeTableWrites.Journal.sequence] using pubAct
      obtain ⟨⟨inv,pc⟩,atMod,actEq⟩ := Option.map_eq_some_iff.mp (List.getElem?_map .. ▸ pubMap)
      have loc := (NativeTableWrites.Journal.mem_mods.mp (List.mem_of_getElem? atMod)).2.2.2
      cases pc with
      | cas => simp [NativeTableWrites.Journal.Site.action] at actEq
      | fence | field _ => cases loc
      | finish =>
        cases actEq
        rw [payloadEq]
        refine ⟨.fromCurrent (2 * (m + 1) - 1) inv atMod,?_⟩
        -- Reuse the session validity proof from `JournalResident`:
        have prior := Nested.toPrefix current.exactRun
        have frame := (CausalTable.certified current.memory current.init prior RootCausal.preserves
          (fun x => (curValid x).1)).1
        have words := SeedHistory.stores current.levels (fun x => (curValid x).2.1)
        apply Table.publication_invariant current.memory a (current.init a)
          (JournalResident.Valid mc fc endian) (curValid a) _ (2 * (m + 1) - 1) inv atMod
        intro j listed addr success previous
        subst a
        have scores : CausalTable.Correct mc fc
            (CausalTable.chosen current.before current.exactFrames ((current.calls j).event 5)) := by
          intro pos pv alpha beta v win narrow chosen
          obtain ⟨f,mem,_,rfl,rfl,rfl,rfl,rfl⟩ := chosen
          exact frame f mem win narrow
        exact ⟨Table.Produced.safe ((prior.store j listed success).mapInputs fun _ _ src => src.safe scores)
            (fun p hSeen => (previous p hSeen).1),
          FiniteTable.written (current.levels.finiteStores.operands j listed success)
            (fun p hSeen => (previous p hSeen).2.1),
          FiniteTable.fieldWritten (words.operands j listed success)
            (fun _ _ range => range) (fun p hSeen => (previous p hSeen).2.2)⟩
  refine ⟨atEpoch.1,atEpoch.2.1,?_⟩
  -- Link the observed `EntryData` `e` at epoch `k` to `Table.Sound mc fc b e`.
  have matchPayload := seen.1
  have safeEpoch : Table.Safe mc fc endian ((current.memory.slot a (current.init a)).payloadAt k) := atEpoch.2.1
  have payloadEq : (current.memory.slot a (current.init a)).payloadAt k =
      ⟨b.val.player,b.val.opponent,Packing.pack endian e⟩ := by
    -- `seen.1` is `Table.Seen ... ⟨b.val.player, b.val.opponent, Packing.pack endian e⟩`,
    -- wait: let's prove `payloadAt k` matches `e` directly from `seen.2`!
    obtain ⟨lastIdx,pIdx,oIdx,mIdx,firstEven⟩ := seen.2.indices
    have hP : (current.memory.memory a).writeValue (.payload .player) k = b.val.player := by
      have rf := seen.2.player
      rw [pIdx] at rf
      have eq := (current.memory.memory a).rf_value rf.1
      rw [rf.2] at eq
      exact eq.symm
    have hO : (current.memory.memory a).writeValue (.payload .opponent) k = b.val.opponent := by
      have rf := seen.2.opponent
      rw [oIdx] at rf
      have eq := (current.memory.memory a).rf_value rf.1
      rw [rf.2] at eq
      exact eq.symm
    have hM : (current.memory.memory a).writeValue (.payload .metadata) k = Packing.pack endian e := by
      have rf := seen.2.metadata
      rw [mIdx] at rf
      have eq := (current.memory.memory a).rf_value rf.1
      rw [rf.2] at eq
      exact eq.symm
    simp only [WriterTrace.Slot.payloadAt,hP,hO,hM]
  rw [payloadEq] at safeEpoch
  have unpackEq : Packing.unpack endian (Packing.pack endian e) = e := Packing.unpack_pack endian e
  have sound := safeEpoch b rfl (by change (Packing.unpack endian (Packing.pack endian e)).bound ≠ 0; rw [unpackEq]; exact seen.1.2)
  rwa [unpackEq] at sound

/-- Compose an arbitrary finite sequence of per-worker cache session timelines on
    the same persistent worker pool. -/
def combineTimelines (p : Pool) : List (Timeline p) → Timeline p
  | [] => ⟨[],by simp⟩
  | [t] => t
  | t :: rest => JournalLocalCache.sessions t (combineTimelines p rest)

/-- Iteratively resume a persistent pool header across an arbitrary list of prior
    session tapes, preserving every frame sample from pool creation onward. -/
def resumeMany (p : Pool) : List (Joint.Tape Unit) → Joint.Tape Op → Option (Joint.Tape Op)
  | [], current =>
    if JournalBodies.noPoolHeader current then some (.poolHistory p.base p.size [] :: current) else none
  | [t], current => JournalBodies.resumePool p t current
  | t :: rest, current => do
    let prev ← resumeMany p rest current
    JournalBodies.resumePool p t prev

/-- Multi-session per-worker local cache carry theorem (shallow + deep caches):
    1. Across any finite list of prior sessions `past` followed by `current`,
       replaying from pool creation agrees with carrying the warm per-worker
       image from `combineTimelines p past` into `current` and satisfies `Good`,
    2. Prepending a single prior session tape via `JournalBodies.resumePool`
       reproduces the exact concatenated frame stream `EntryStream.capture`. -/
def LocalCacheChainContract : Prop :=
  ∀ (p : Pool) (past : List (Timeline p)) (current : Timeline p)
    (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (worker step : Nat),
    let prev := combineTimelines p past
    (JournalLocalCache.sessions prev current).readAt mc fc worker (JournalLocalCache.offset prev + step) =
      JournalLocalCache.readSessions prev current mc fc worker step ∧
    Good mc fc p.width p.sizes p.key (JournalLocalCache.readSessions prev current mc fc worker step)

theorem localCache_chain : LocalCacheChainContract := by
  intro p past current mc fc worker step
  exact JournalLocalCache.carries.2 p (combineTimelines p past) current mc fc worker step

/-- Combined warm engine pool state carried into a new search request:
    both the shared TT (chained through drained sessions with `Entry.Resident`)
    and the persistent worker pool's local endgame caches (`Good`). -/
structure WarmPool (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (endian : Bool) where
  pool : Pool
  firstSession : JournalResident.Session mc fc endian
  lastSession : JournalResident.Session mc fc endian
  seedOrigins : ∀ a, SlotOrigin mc fc endian (firstSession.initial a)
  ttChain : JournalResident.Chain firstSession lastSession
  pastCaches : List (Timeline pool)

theorem WarmPool.readyFor (w : WarmPool mc fc endian)
    (current : JournalResident.Session mc fc endian)
    (handoff : JournalResident.Handoff w.lastSession current.memory)
    (curCache : Timeline w.pool) :
    (∀ a, Even (current.memory.memory a) ∧ Entry.Resident mc fc endian (current.initial a)) ∧
    (∀ worker step, Good mc fc w.pool.width w.pool.sizes w.pool.key
      (JournalLocalCache.readSessions (combineTimelines w.pool w.pastCaches) curCache mc fc worker step)) :=
  ⟨fun a => let h := (chain_resident mc fc endian w.firstSession w.lastSession current w.seedOrigins w.ttChain handoff).1 a
            ⟨h.1,h.2.1⟩,
   fun worker step => (localCache_chain w.pool w.pastCaches curCache mc fc worker step).2⟩

end NeuralReversi.PvsParallel.Interleaving.MultiSessionCarry
