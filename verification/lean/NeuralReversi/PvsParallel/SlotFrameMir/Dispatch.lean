import NeuralReversi.PvsParallel.SlotFrameMir.DispatchProgram

namespace NeuralReversi.PvsParallel.SlotFrameMir

open Threading
open AtomicWord (Event Read Write SequencedBefore ReadWindow InReleaseSequence)
open CutoffVisibility (History HappensBefore)

inductive IdlePost
  | split (slot : Slot) (flag : Event)
  | helper (put flag : Event)

def IdlePost.flag : IdlePost → Write
  | .split _ e | .helper _ e => ⟨e,1,true,false⟩

def activeAfter (initial : Pointer) : List IdlePost → Pointer
  | [] => initial
  | .split s _::xs => activeAfter (.pending s) xs
  | .helper .. ::xs => activeAfter initial xs

def Pointer.address : Pointer → Option Slot
  | .empty => none | .pending s | .used s => some s

private theorem pending_end {initial : Pointer} (posts : List IdlePost)
    (base : ∃ s, initial=.pending s) : ∃ s, activeAfter initial posts=.pending s := by
  induction posts generalizing initial with
  | nil => exact base
  | cons p xs ih => cases p with
    | split s _ => exact ih ⟨s,rfl⟩
    | helper _ _ => exact ih base

private theorem any_split {initial : Pointer} (posts : List IdlePost) (s : Slot) (e : Event)
    (loc : .split s e ∈ posts) : ∃ target, activeAfter initial posts=.pending target := by
  induction posts generalizing initial with
  | nil => cases loc
  | cons p xs ih =>
    cases p with
    | split w _ => exact pending_end xs ⟨w,rfl⟩
    | helper _ _ => exact ih (List.mem_cons.mp loc |>.resolve_left (by intro eq;cases eq))

private theorem clean_end (initial : Pointer) (posts : List IdlePost)
    (good : ∀ s, initial ≠ .used s) : ∀ s, activeAfter initial posts ≠ .used s := by
  induction posts generalizing initial with
  | nil => exact good
  | cons p xs ih => cases p with
    | split s _ => exact ih _ (by simp)
    | helper _ _ => exact ih _ good

/-- The searching-word interval starts at the previous helper return's false
    store. It ends at the active-pointer snapshot. All intervening external
    searching stores are split or helper postings under the target thread lock. -/
structure HelperFlag (history : History) (posts : List IdlePost) where
  earlier : List Write
  completion : Event
  later : List Write
  load : Read (earlier ++ (⟨completion,if completedHelperBusy then 1 else 0,true,false⟩ :: posts.map IdlePost.flag) ++ later)
  window : ReadWindow load (HappensBefore history) ⟨earlier.length,by simp⟩ (earlier.length+posts.length)
  acquire : load.acquire=true
  sync : ∀ i, load.acquire=true → InReleaseSequence _ i load.source →
    history.otherSync ((earlier ++ (⟨completion,if completedHelperBusy then 1 else 0,true,false⟩ :: posts.map IdlePost.flag) ++ later).get i).event load.event

def HelperFlag.value (f : HelperFlag h posts) : BitVec 64 :=
  ((f.earlier ++ (⟨f.completion,if completedHelperBusy then 1 else 0,true,false⟩ :: posts.map IdlePost.flag) ++ f.later).get f.load.source).value

theorem HelperFlag.selected (f : HelperFlag hist posts) (on : f.value=1) :
    ∃ p ∈ posts, HappensBefore hist p.flag.event f.load.event := by
  let span := f.earlier ++ ((⟨f.completion,if completedHelperBusy then 1 else 0,true,false⟩ : Write) :: posts.map IdlePost.flag)
  let words := span ++ f.later
  let i := f.load.source
  have bounds:=f.window.bounds
  have inside : i.val < span.length := by
    have top:=bounds.2;change f.load.source.val≤_ at top;simp [span,i];omega
  have low : f.earlier.length ≤ i.val := bounds.1
  have got : words[i.val]? = some (words.get i) := List.getElem?_eq_getElem _
  change (span++f.later)[i.val]? = _ at got
  rw [List.getElem?_append_left inside] at got
  dsimp only [span] at got
  rw [List.getElem?_append_right low] at got
  cases eq : i.val - f.earlier.length with
  | zero =>
    rw [eq] at got
    have same : (⟨f.completion,0,true,false⟩ : Write) = words.get i := Option.some.inj got
    change (words.get i).value = 1 at on
    rw [← same] at on;exact ((by decide : (0:BitVec 64) ≠ 1) on).elim
  | succ n =>
    rw [eq] at got
    change (posts.map IdlePost.flag)[n]? = _ at got
    rw [List.getElem?_map] at got
    obtain ⟨p,where_,same⟩:=Option.map_eq_some_iff.mp got
    have rel : (words.get i).release=true := by rw [← same];cases p <;> rfl
    have seq : InReleaseSequence words i i := ⟨rel,Nat.le_refl _,fun _ lt le => by omega⟩
    have causal : HappensBefore hist (words.get i).event f.load.event := .other (f.sync i f.acquire seq)
    exact ⟨p,List.mem_of_getElem? where_,same ▸ causal⟩

/-- Mutex order of all helper job puts between the preceding scheduler callback
    and the next take. A put after this take is ordered by the job mutex. -/
structure JobTake (hist : History) (posts : List IdlePost) where
  initial : Bool
  puts : List Event
  taken : Event
  mutex : ∀ put flag, IdlePost.helper put flag ∈ posts →
    put ∈ puts ∨ HappensBefore hist taken put
  posted : ∀ put flag, IdlePost.helper put flag ∈ posts → SequencedBefore put flag

def JobTake.someJob (j : JobTake hist posts) : Bool :=
  match j.puts with | [] => j.initial | _::_ => postHelperJob

inductive IdleEntry (hist : History) (start : Pointer) (posts : List IdlePost) : Type
  | born : start=.empty → IdleEntry hist start posts
  | owned (s : Slot) : start=.pending s → IdleEntry hist start posts
  | priorWork : start=.empty → IdleEntry hist start posts
  | priorHelper (f : HelperFlag hist posts) (j : JobTake hist posts) :
    dispatchWork (decide (f.value=1)) j.someJob = true →
    SequencedBefore f.load.event j.taken → IdleEntry hist start posts

/-- A normal work call that successfully extracts Some(active) cannot recapture a
    pointer restored by finalize. This includes arbitrary helper-job posts, whose
    busy store is preceded by a Some(job) write. -/
theorem dispatch_fresh (origin : IdleEntry hist initial posts) (target : Slot)
    (read : (activeAfter initial posts).address=some target)
    (acyclic : ∀ e, ¬HappensBefore hist e e) : activeAfter initial posts=.pending target := by
  suffices safe : activeAfter initial posts ≠ .used target by
    cases eq : activeAfter initial posts with
    | empty => rw [eq] at read;cases read
    | used s => rw [eq] at read;cases read;exact (safe eq).elim
    | pending s => rw [eq] at read;cases read;rfl
  cases origin with
  | born start => subst initial;exact clean_end _ posts (by simp) target
  | owned s start => subst initial;exact clean_end _ posts (by simp) target
  | priorWork start => subst initial;exact clean_end _ posts (by simp) target
  | priorHelper f job yes before =>
    obtain ⟨busy,empty⟩ := Bool.and_eq_true_iff.mp yes
    obtain ⟨source,mem,synced⟩ := f.selected (of_decide_eq_true busy)
    cases source with
    | split s e =>
      obtain ⟨v,known⟩:=any_split (initial:=initial) posts s e mem
      rw [known];simp
    | helper put flag =>
      have ahead := HappensBefore.trans (.sequenced (job.posted put flag mem)) (synced.trans (.sequenced before))
      rcases job.mutex put flag mem with occurred | after
      · cases eq : job.puts with
        | nil => rw [eq] at occurred;cases occurred
        | cons => simp [JobTake.someJob,eq,postHelperJob] at empty
      · exact (acyclic put (ahead.trans after)).elim

end NeuralReversi.PvsParallel.SlotFrameMir
