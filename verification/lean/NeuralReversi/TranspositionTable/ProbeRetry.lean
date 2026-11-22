import NeuralReversi.TranspositionTable.Replacement

namespace NeuralReversi.TranspositionTable.ProbeRetry

inductive Observation where
  | busy
  | empty
  | matching (data : EntryData)
  | other (data : EntryData)

structure Attempt where
  first : Observation
  second : Observation
  generation : BitVec 8

inductive Result where
  | hit (slot : Fin 2) (data : EntryData)
  | miss (slot : Fin 2)

def scan (fallback : Fin 2) : List Attempt → Result
  | [] => .miss fallback
  | a :: rest =>
    match a.first, a.second with
    | .empty, _ => .miss 0
    | .matching data, _ => .hit 0 data
    | _, .empty => .miss 1
    | _, .matching data => .hit 1 data
    | .other x, .other y => .miss (Replacement.victim a.generation x y)
    | .busy, .other _ => scan 1 rest
    | .other _, .busy => scan 0 rest
    | .busy, .busy => scan fallback rest

def probe (attempts : List Attempt) : Result :=
  scan 0 (attempts.take 3)

private def slotResult (slot : Fin 2) : Observation → Option Result
  | .empty => some (.miss slot)
  | .matching data => some (.hit slot data)
  | _ => none

def decisive (a : Attempt) : Option Result :=
  (slotResult 0 a.first).or ((slotResult 1 a.second).or (match a.first, a.second with
    | .other x, .other y => some (.miss (Replacement.victim a.generation x y))
    | _, _ => none))

def candidate (a : Attempt) : Option (Fin 2) :=
  match a.first, a.second with
  | .other _, .busy => some 0
  | .busy, .other _ => some 1
  | _, _ => none

private theorem scan_spec (attempts : List Attempt) (fallback : Fin 2) :
    scan fallback attempts = (attempts.findSome? decisive).getD
      (.miss ((attempts.reverse.findSome? candidate).getD fallback)) := by
  induction attempts generalizing fallback with
  | nil => rfl
  | cons a rest ih =>
    obtain ⟨first, second, gen⟩ := a
    cases first <;> cases second <;>
      simp [scan, List.findSome?_cons, List.findSome?_append, decisive, slotResult, candidate, ih,
        Option.getD, Option.or] <;>
      cases rest.reverse.findSome? candidate <;> rfl

abbrev Key := BitVec 64 × BitVec 64

structure RawAttempt where
  readSlot : Fin 2 → Option (Key × EntryData)
  generation : BitVec 8

def classify (key : Key) : Option (Key × EntryData) → Observation
  | none => .busy
  | some (stored, data) =>
    if data.bound = 0 then .empty
    else if sameBoardBits stored.1 stored.2 key.1 key.2 then .matching data else .other data

def observe (key : Key) (a : RawAttempt) : Attempt :=
  ⟨classify key (a.readSlot 0), classify key (a.readSlot 1), a.generation⟩

private theorem matching_source {key : Key} {raw : Option (Key × EntryData)} {d : EntryData}
    (h : classify key raw = .matching d) : raw = some (key, d) ∧ d.bound ≠ 0 := by
  cases raw with
  | none => contradiction
  | some r =>
    obtain ⟨stored, data⟩ := r
    simp only [classify] at h
    split at h
    · contradiction
    · split at h
      · rename_i equal
        obtain ⟨p, o⟩ := sameBoardBits_exact _ _ _ _ |>.mp equal
        cases Prod.ext p o
        cases h
        exact ⟨rfl, by assumption⟩
      · contradiction

private theorem decisive_hit (a : Attempt) (slot : Fin 2) (d : EntryData)
    (h : decisive a = some (.hit slot d)) :
    (slot = 0 ∧ a.first = .matching d) ∨ (slot = 1 ∧ a.second = .matching d) := by
  obtain ⟨first, second, gen⟩ := a
  cases first <;> cases second <;> simp_all [decisive, slotResult, Option.or]

def HitProvenance : Prop :=
  ∀ key (attempts : List RawAttempt) slot data,
    probe (attempts.map (observe key)) = .hit slot data →
    ∃ a ∈ attempts.take 3, a.readSlot slot = some (key, data) ∧ data.bound ≠ 0

private theorem hit_provenance : HitProvenance := by
  intro key attempts slot data result
  rw [probe, scan_spec, ← List.map_take] at result
  cases found : ((attempts.take 3).map (observe key)).findSome? decisive with
  | none => rw [found] at result; contradiction
  | some r =>
    rw [found] at result
    change r = .hit slot data at result
    subst r
    obtain ⟨a, member, selected⟩ := List.exists_of_findSome?_eq_some found
    obtain ⟨raw, mem, rfl⟩ := List.mem_map.mp member
    refine ⟨raw, mem, ?_⟩
    rcases decisive_hit _ _ _ selected with ⟨rfl, h⟩ | ⟨rfl, h⟩ <;>
      exact matching_source h

def Contract : Prop :=
  (∀ attempts, let observed := attempts.take 3
    probe attempts = (observed.findSome? decisive).getD
      (.miss ((observed.reverse.findSome? candidate).getD 0))) ∧ HitProvenance

theorem contract_proven : Contract :=
  ⟨fun attempts => scan_spec (attempts.take 3) 0, hit_provenance⟩

end NeuralReversi.TranspositionTable.ProbeRetry
