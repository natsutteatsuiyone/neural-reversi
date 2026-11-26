import NeuralReversi.PvsParallel.FrameTableReads.Rounds
import NeuralReversi.PvsParallel.NativeLeafBranches

namespace NeuralReversi.PvsParallel.FrameTableReads.Operations

open TranspositionTable
open SeqLock (Word)
open FrameReturns (Log Frame)

/-- Completed MIR iterations construct decoder packets until the same return
    or the same exhausted retry budget, including a carried fallback victim. -/
private theorem sweep_rounds {log : Log mc fc} {t : Table log} {f : Frame} (m : Machine t (log.identify f.use) issued s)
    (allocation : Nat) (clusters key : Word) (endian : Bool) (query : ProbeRetry.Key) (fuel start : Nat) (last : Fin 2) :
    let execution := sweep s allocation clusters key endian query fuel start last
    Runs issued start (execution.emit s endian query) →
    ∃ rows decoded, rounds t allocation clusters key f endian query fuel rows = some decoded ∧
      decoded.map ProbeRound.raw = execution.rounds.map (fun r => r.read s endian) ∧ rows.length = execution.rounds.length := by
  induction fuel generalizing start last with
  | zero =>
    intro _ _
    exact ⟨[],[],rfl,rfl,rfl⟩
  | succ fuel ih =>
    let r := roundAt s allocation clusters key endian query start
    let obs := ProbeRetry.observe query (r.read s endian)
    have decision : (r.pass s endian query).result = ProbeRetry.decisive obs := ReaderMir.pass_decisive obs
    simp only [sweep,←show r = roundAt s allocation clusters key endian query start from rfl]
    cases branch : (r.pass s endian query).exit with
    | done result =>
      intro ran
      have one : Runs issued start (r.emit s endian query) := by simpa [Probed.emit] using ran
      obtain ⟨packets,a,read,raw⟩ := r.decode m (placed ..) one
      have stopped : ProbeRetry.decisive (ProbeRetry.observe query a.raw) = some result := by
        rw [raw]; exact decision.symm.trans (by simp [ReaderMir.Pass.result,branch])
      exact ⟨[(r.generation,packets)],[a],by simp [rounds,read,stopped],by simp [raw],rfl⟩
    | retry choice =>
      intro ran
      let next := start+(r.emit s endian query).length
      let tail := sweep s allocation clusters key endian query fuel next (choice.getD last)
      have whole : Runs issued start (r.emit s endian query ++ tail.emit s endian query) := ran
      obtain ⟨one,more⟩ := whole.append
      obtain ⟨packets,a,read,raw⟩ := r.decode m (placed ..) one
      obtain ⟨parts,parsed,recursed,raws,count⟩ := ih next (choice.getD last) more
      have retry : ProbeRetry.decisive (ProbeRetry.observe query a.raw) = none := by
        rw [raw]; exact decision.symm.trans (by simp [ReaderMir.Pass.result,branch])
      exact ⟨(r.generation,packets)::parts,a::parsed,by simp [rounds,read,retry,recursed],
        List.cons.injEq .. |>.mpr ⟨raw,raws⟩,congrArg Nat.succ count⟩

theorem probe_packets {log : Log mc fc} {t : Table log} {f : Frame} (m : Machine t (log.identify f.use) issued s)
    (allocation : Nat) (clusters key : Word) (endian : Bool) (query : ProbeRetry.Key) (start : Nat) :
    let execution := probe s allocation clusters key endian query start
    Runs issued start (execution.emit s endian query) →
    ∃ packets attempts, decodeProbe t allocation clusters key f endian query packets = some attempts ∧
      attempts = execution.rounds.map (fun r => r.read s endian) ∧
      ProbeRetry.probe (attempts.map (ProbeRetry.observe query)) = execution.result ∧
      TableReads.ProbeReadsFrom (FrameReturns.Samples log endian f) attempts := by
  intro execution ran
  obtain ⟨packets,decoded,read,raws,_⟩ := sweep_rounds m allocation clusters key endian query ReaderMir.probeRetries start 0 ran
  have accepted : decodeProbe t allocation clusters key f endian query packets = some (decoded.map ProbeRound.raw) := by simp [decodeProbe,read]
  exact ⟨packets,_,accepted,raws,by rw [raws,List.map_map]; exact probe_result ..,decoded_probe_reads t accepted⟩

private theorem round_origin {log : Log mc fc} {t : Table log} (r : ProbeRound t allocation clusters key f endian query)
    (hit : r.raw.readSlot slot = some (query,d)) (occupied : d.bound ≠ 0#8) :
    Published log (log.identify f.use) endian query d := by
  unfold ProbeRound.raw at hit
  dsimp only at hit
  split at hit
  · exact snapshot_published r.front hit occupied
  · obtain ⟨b,_,hit⟩ := Option.bind_eq_some_iff.mp hit
    exact snapshot_published b hit occupied

theorem probe_published {log : Log mc fc} {t : Table log} {f : Frame} (m : Machine t (log.identify f.use) issued s)
    (allocation : Nat) (clusters key : Word) (endian : Bool) (query : ProbeRetry.Key) (start : Nat) :
    let execution := probe s allocation clusters key endian query start
    Runs issued start (execution.emit s endian query) → execution.result = .hit slot d →
    Published log (log.identify f.use) endian query d := by
  intro execution ran hit
  obtain ⟨_,decoded,_,raws,_⟩ := sweep_rounds m allocation clusters key endian query ReaderMir.probeRetries start 0 ran
  have probed : ProbeRetry.probe ((decoded.map ProbeRound.raw).map (ProbeRetry.observe query)) = .hit slot d := by
    rw [raws,List.map_map]
    exact (probe_result ..).trans hit
  obtain ⟨a,mem,selected,occupied⟩ := ProbeRetry.contract_proven.2 query _ slot d probed
  obtain ⟨actual,_,eq⟩ := List.mem_map.mp (List.mem_of_mem_take mem)
  exact round_origin actual (eq ▸ selected) occupied

/-- Supply the actual optional probe of a twelve-empty parallel NonPV leaf
    from the executed slot instructions at that frame. -/
theorem leaf_probe {log : Log mc fc} {t : Table log} {r : NativeFrames.Row} (m : Machine t (log.identify r.frame.use) issued s)
    (input : EndgameSearch.CacheHistory.Input) (depth : input.depth.toNat = 12) (parallel : input.parallel = true)
    (allocation : Nat) (clusters key : Word) (endian : Bool) (start : Nat) :
    let query := Cutoffs.key input.board
    let execution := probe s allocation clusters key endian query start
    Runs issued start (execution.emit s endian query) →
    ∃ packet, NativeLeafBranches.probe {input with attempts := execution.rounds.map (fun a => a.read s endian)}
      (some (packet : NativeLeafBranches.Probe t r endian)) = true := by
  intro query execution ran
  obtain ⟨packets,_,accepted,rfl,_,_⟩ := probe_packets m allocation clusters key endian query start ran
  exact ⟨⟨allocation,clusters,key,packets⟩,by simp [NativeLeafBranches.probe,depth,parallel,accepted,query,execution]⟩

end NeuralReversi.PvsParallel.FrameTableReads.Operations
