module

@[expose] public section

namespace NeuralReversi.Threading.SplitCutoff

abbrev Epoch := BitVec 64

structure Cache where
  seen : Epoch
  latched : Bool

structure Observation where
  epoch : Epoch
  chain : List Bool

def reset (epoch : Epoch) : Cache :=
  ⟨epoch - 1, false⟩

def poll (cache : Cache) (read : Observation) : Cache :=
  if cache.latched || read.epoch == cache.seen then cache
  else ⟨read.epoch, read.chain.any id⟩

def mark (previous : Bool) : Bool × Bool :=
  (!previous, true)

def publish (first : Bool) (epoch : Epoch) : Epoch :=
  if first then epoch + 1 else epoch

def MarkContract : Prop :=
  ∀ (previous : Bool) (epoch : Epoch),
    (mark previous).2 = true ∧
    ((mark previous).1 = true ↔ previous = false) ∧
    (publish (mark previous).1 epoch).toNat =
      (epoch.toNat + (if previous then 0 else 1)) % 2^64

theorem markContract_proven : MarkContract := by
  intro previous epoch
  cases previous <;> simp [mark, publish, BitVec.toNat_add, Nat.mod_eq_of_lt epoch.isLt]

def PollContract : Prop :=
  ∀ (cache : Cache) (read : Observation),
    (cache.latched = true → poll cache read = cache) ∧
    (read.epoch = cache.seen → poll cache read = cache) ∧
    (cache.latched = false → read.epoch ≠ cache.seen →
      (poll cache read).seen = read.epoch ∧
      ((poll cache read).latched = true ↔ true ∈ read.chain))

theorem pollContract_proven : PollContract := by
  intro cache read
  have scan : read.chain.any id = true ↔ true ∈ read.chain := by
    rw [List.any_eq_true]
    constructor
    · rintro ⟨flag, member, rfl⟩
      exact member
    · intro member
      exact ⟨true, member, rfl⟩
  exact ⟨fun latched => by simp [poll, latched], fun same => by simp [poll, same],
    fun fresh changed => by simp [poll, fresh, changed, scan]⟩

private theorem reset_alias (epoch : Epoch) (n : Nat) :
    epoch + BitVec.ofNat 64 n = (reset epoch).seen ↔ (n + 1) % 2^64 = 0 := by
  constructor
  · intro alias
    have plus := congrArg (fun e : Epoch => e + 1) alias
    simp only [reset, BitVec.sub_add_cancel, BitVec.add_assoc] at plus
    have bumped := congrArg (fun e : Epoch => e - epoch) plus
    rw [BitVec.add_comm epoch, BitVec.add_sub_cancel, BitVec.sub_self] at bumped
    have wrapped := congrArg BitVec.toNat bumped
    simpa [BitVec.toNat_add, Nat.mod_add_mod] using wrapped
  · intro alias
    have bumped : BitVec.ofNat 64 n + 1 = 0 := by
      apply BitVec.eq_of_toNat_eq
      simpa [BitVec.toNat_add, Nat.mod_add_mod] using alias
    have plus : epoch + BitVec.ofNat 64 n + 1 = epoch := by
      rw [BitVec.add_assoc, bumped]
      simp
    calc
      epoch + BitVec.ofNat 64 n = (epoch + BitVec.ofNat 64 n + 1) - 1 := by
        rw [BitVec.add_sub_cancel]
      _ = (reset epoch).seen := congrArg (fun e : Epoch => e - 1) plus

def ResetContract : Prop :=
  ∀ (epoch : Epoch) (increments : Nat) (read : Observation),
    read.epoch = epoch + BitVec.ofNat 64 increments →
    (read.epoch = (reset epoch).seen ↔ (increments + 1) % 2^64 = 0) ∧
    ((increments + 1) % 2^64 ≠ 0 →
      (poll (reset epoch) read).seen = read.epoch ∧
      ((poll (reset epoch) read).latched = true ↔ true ∈ read.chain))

theorem resetContract_proven : ResetContract := by
  intro epoch increments read observed
  have alias := observed ▸ reset_alias epoch increments
  exact ⟨alias, fun distinct =>
    (pollContract_proven (reset epoch) read).2.2 rfl (fun eq => distinct (alias.mp eq))⟩

theorem epoch_index_eq (previous current : Nat) (ordered : previous ≤ current)
    (withinCycle : current - previous < 2^64)
    (words : BitVec.ofNat 64 current = BitVec.ofNat 64 previous) : current = previous := by
  have sum : BitVec.ofNat 64 previous + BitVec.ofNat 64 (current - previous) =
      BitVec.ofNat 64 previous := by
    rw [← BitVec.ofNat_add, Nat.add_sub_of_le ordered, words]
  have diff := congrArg (fun w : Epoch => w - BitVec.ofNat 64 previous) sum
  rw [BitVec.add_comm, BitVec.add_sub_cancel, BitVec.sub_self] at diff
  have zero := congrArg BitVec.toNat diff
  simp [Nat.mod_eq_of_lt withinCycle] at zero
  omega

def VisibilityContract : Prop :=
  ∀ (cache : Cache) (read : Observation) (previous current published : Nat),
    cache.seen = BitVec.ofNat 64 previous →
    (cache.latched = false → previous < published) →
    read.epoch = BitVec.ofNat 64 current →
    previous ≤ current → current - previous < 2^64 →
    (published ≤ current → true ∈ read.chain) →
    let next := poll cache read
    (published ≤ current → next.latched = true) ∧
    (next.latched = false → next.seen = BitVec.ofNat 64 current ∧ current < published)

theorem visibilityContract_proven : VisibilityContract := by
  intro cache read previous current published seen miss epoch ordered withinCycle visible
  obtain ⟨latched, hit, refresh⟩ := pollContract_proven cache read
  cases prior : cache.latched with
  | true => simp [latched prior, prior]
  | false =>
    by_cases same : read.epoch = cache.seen
    · have sameCount := epoch_index_eq previous current ordered withinCycle (epoch.symm.trans (same.trans seen))
      have before := miss prior
      rw [hit same]
      exact ⟨fun entered => by omega, fun _ => ⟨same.symm.trans epoch, by omega⟩⟩
    · obtain ⟨updated, cutoff⟩ := refresh prior same
      exact ⟨fun entered => cutoff.mpr (visible entered), fun clear =>
        ⟨updated.trans epoch, by
          by_cases entered : published ≤ current
          · simp [cutoff.mpr (visible entered)] at clear
          · omega⟩⟩

structure Polling where
  cache : Cache
  observations : Nat → Observation
  cursor : Nat

def Polling.after (state : Polling) : Polling :=
  if state.cache.latched then state
  else { state with cache := poll state.cache (state.observations state.cursor), cursor := state.cursor + 1 }

def Polling.occurred (state : Polling) : Bool :=
  state.after.cache.latched

theorem after_latched {state : Polling} (latched : state.cache.latched = true) :
    state.after = state := by simp [Polling.after, latched]

theorem occurred_latched {state : Polling} (latched : state.cache.latched = true) :
    state.occurred = true := by simpa [Polling.occurred, after_latched latched] using latched

def trace : Nat → Polling → List Bool
  | 0, _ => []
  | n + 1, state => state.occurred :: trace n state.after

private theorem latched_trace (n : Nat) (state : Polling) (latched : state.cache.latched = true) :
    trace n state = List.replicate n true := by
  induction n with
  | zero => rfl
  | succ n ih => simp [trace, occurred_latched latched, after_latched latched, ih, List.replicate_succ]

def TraceContract : Prop :=
  ∀ (n : Nat) (state : Polling),
    ∃ before after, before + after = n ∧
      trace n state = List.replicate before false ++ List.replicate after true

theorem traceContract_proven : TraceContract := by
  intro n state
  induction n generalizing state with
  | zero => exact ⟨0, 0, rfl, rfl⟩
  | succ n ih =>
    cases observed : state.occurred with
    | false =>
      obtain ⟨before, after, count, polls⟩ := ih state.after
      exact ⟨before + 1, after, by omega, by simp [trace, observed, polls, List.replicate_succ]⟩
    | true =>
      exact ⟨0, n + 1, by simp,
        by simp [trace, observed, latched_trace n state.after observed, List.replicate_succ]⟩

end NeuralReversi.Threading.SplitCutoff
