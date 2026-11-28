module
public import NeuralReversi.PvsParallel.FrameReturns
public import NeuralReversi.PvsParallel.RootOwnerMir.Slice.Calls

@[expose] public section

namespace NeuralReversi.PvsParallel.FrameReturns.OwnerCalls

open PvsBoardTree (GeneratedSquare play)
open RootOwnerMir.Slice (Visit Path Supplies)

/-- The operands and result of a normal direct child, in the child's perspective. -/
structure ChildReturn where
  board : Board.Board
  depth : Nat
  pv : Bool
  cut : Bool
  alpha : Int
  beta : Int
  score : Int
  deriving DecidableEq

def direct (owner child : Frame) : Bool :=
  child.returnStep.any fun stamp =>
    decide (child.parent = owner.id ∧ child.worker = owner.worker ∧ stamp < owner.scoreStep)

def fields (child : Frame) : ChildReturn :=
  ⟨child.toFrameReturn.board,child.depth,child.pv,child.cut,child.alpha,child.beta,child.score⟩

def observed (log : Log mc fc) (f : Frame) : List ChildReturn :=
  (log.frames.filter (direct f)).map fields

variable {mc : Bitboard.Moves.X86Config} {fc : Cutoffs.FlipConfig}

def child (f : Frame) (d : TranspositionTable.Depth) (v : Visit (GeneratedSquare mc f.position.val))
    (pv : Bool) (word : Int) : ChildReturn :=
  ⟨(play mc fc f.position v.move).val.val,(d-1).toNat,pv,!pv && !f.cut,v.lower pv f.beta, -v.alpha,word⟩

def emitted (fc : Cutoffs.FlipConfig) (f : Frame) (depth : TranspositionTable.Depth)
    (visits : List (Visit (GeneratedSquare mc f.position.val))) : List ChildReturn :=
  visits.flatMap fun v => [false,true].filterMap fun pv => (v.word pv).map (child (mc:=mc) (fc:=fc) f depth v pv)

theorem emitted_iff {f : Frame} {row : ChildReturn} {d : TranspositionTable.Depth}
    {visits : List (Visit (GeneratedSquare mc f.position.val))} :
    row ∈ emitted (mc:=mc) fc f d visits ↔ ∃ v ∈ visits, ∃ pv word, v.word pv = some word ∧ child (mc:=mc) (fc:=fc) f d v pv word = row := by
  constructor
  · intro h
    obtain ⟨v,mem,found⟩ := List.mem_flatMap.mp h
    obtain ⟨pv,_,got⟩ := List.mem_filterMap.mp found
    obtain ⟨word,wordAt,eq⟩ := Option.map_eq_some_iff.mp got
    exact ⟨v,mem,pv,word,wordAt,eq⟩
  · rintro ⟨v,mem,pv,word,wordAt,rfl⟩
    exact List.mem_flatMap.mpr ⟨v,mem,List.mem_filterMap.mpr ⟨pv,by cases pv <;> simp,by simp [wordAt]⟩⟩

private theorem matches_fields (f g : Frame) (b : PvsBoardTree.Position) (d : TranspositionTable.Depth)
    (pv cut : Bool) (a z : Int) :
    matchesCall g f.id f.use b d pv cut a z = true ↔ direct f g = true ∧
      fields g = ⟨b.val,d.toNat,pv,cut,a,z,g.score⟩ := by
  cases h : g.returnStep <;> simp [matchesCall,direct,fields,h,Frame.use,ChildReturn.mk.injEq,and_assoc] <;> exact ⟨of_decide_eq_true,decide_eq_true⟩

def passCall (f : Frame) (d : TranspositionTable.Depth) (pv : Bool) (word : Int) : ChildReturn :=
  ⟨(PvsBoardTree.pass f.position).val,d.toNat,pv,!pv && !f.cut,-f.beta,-f.alpha,word⟩

theorem pass_reply (log : Log mc fc) (f : Frame) (d : TranspositionTable.Depth) (pv : Bool) (word : Int)
    (only : (observed log f).Perm [passCall f d pv word]) :
    log.reply f.id f.use (PvsBoardTree.pass f.position) d pv (!pv && !f.cut) (-f.beta) (-f.alpha) = some word := by
  let test := fun g => matchesCall g f.id f.use (PvsBoardTree.pass f.position) d pv (!pv && !f.cut) (-f.beta) (-f.alpha)
  have present := only.mem_iff.mpr List.mem_cons_self
  obtain ⟨g,seen,eq⟩ := List.mem_map.mp present
  obtain ⟨member,own⟩ := List.mem_filter.mp seen
  have score : g.score = word := congrArg ChildReturn.score eq
  have hit : test g := (matches_fields _ _ _ _ _ _ _ _).mpr ⟨own,score ▸ eq⟩
  change (log.frames.find? test).map (fun g => g.score) = some word
  cases found : log.frames.find? test with
  | none => have := List.find?_eq_none.mp found g member; simp [hit] at this
  | some chosen =>
    have accepted : test chosen = true := List.find?_some found
    have selected := ((matches_fields f chosen _ _ _ _ _ _).mp accepted).1
    have inLog : fields chosen ∈ observed log f := List.mem_map.mpr
      ⟨chosen,List.mem_filter.mpr ⟨List.mem_of_find?_eq_some found,selected⟩,rfl⟩
    have same := List.mem_singleton.mp (only.mem_iff.mp inLog)
    exact congrArg some (congrArg ChildReturn.score same)

private theorem hit_iff (f g : Frame) (v : Visit (GeneratedSquare mc f.position.val))
    (d : TranspositionTable.Depth) (pv : Bool) :
    matchesCall g f.id f.use (play mc fc f.position v.move).val (d-1) pv (!pv && !f.cut) (v.lower pv f.beta) (-v.alpha) = true ↔
      direct f g = true ∧ fields g = child (mc:=mc) (fc:=fc) f d v pv g.score := by
  exact matches_fields _ _ _ _ _ _ _ _

private theorem at_visit {v w : Visit Move} {visits : List (Visit Move)} (unique : (visits.map Visit.move).Nodup)
    (left : v ∈ visits) (right : w ∈ visits) (same : v.move = w.move) : v = w := by
  induction visits with
  | nil => simp at left
  | cons x xs ih =>
    obtain ⟨absent,nodup⟩ := List.nodup_cons.mp unique
    rcases List.mem_cons.mp left with vx | vl
    · subst v
      rcases List.mem_cons.mp right with wx | wr
      · exact wx.symm
      · exact (absent (List.mem_map.mpr ⟨_,wr,same.symm⟩)).elim
    · rcases List.mem_cons.mp right with wx | wr
      · subst w
        exact (absent (List.mem_map.mpr ⟨_,vl,same⟩)).elim
      · exact ih nodup vl wr

/-- Matching the pool's direct child records in any order suffices. Distinct legal
    moves produce distinct child boards, and the scout and wide roles differ even
    when their windows coincide. Other parents and workers cannot shadow the call. -/
theorem supplies (log : Log mc fc) (f : Frame) (d : TranspositionTable.Depth)
    (visits : List (Visit (GeneratedSquare mc f.position.val))) (unique : (visits.map Visit.move).Nodup)
    (records : (observed log f).Perm (emitted fc f d visits)) : Supplies (log.pvReplies f d) f.beta visits := by
  intro v mem pv word returned
  let expected := child (mc:=mc) (fc:=fc) f d v pv word
  have present : expected ∈ observed log f :=
    records.mem_iff.mpr (emitted_iff.mpr ⟨v,mem,pv,word,returned,rfl⟩)
  obtain ⟨g,kept,actual⟩ := List.mem_map.mp present
  obtain ⟨member,local_⟩ := List.mem_filter.mp kept
  let test := fun row => matchesCall row f.id f.use (play mc fc f.position v.move).val (d-1) pv
    (!pv && !f.cut) (v.lower pv f.beta) (-v.alpha)
  have hit : test g = true := by
    have value : g.score = word := congrArg ChildReturn.score actual
    exact (hit_iff (mc:=mc) (fc:=fc) f g v d pv).mpr ⟨local_,value ▸ actual⟩
  change (log.frames.find? test).map (fun g : Frame => g.score) = some word
  cases found : log.frames.find? test with
  | none =>
    have rejected := List.find?_eq_none.mp found g member
    simp [hit] at rejected
  | some selected =>
    have hitSelected : test selected = true := List.find?_some found
    obtain ⟨own,selectedFields⟩ := (hit_iff (mc:=mc) (fc:=fc) f selected v d pv).mp hitSelected
    have inLog : fields selected ∈ observed log f :=
      List.mem_map.mpr ⟨selected,List.mem_filter.mpr ⟨List.mem_of_find?_eq_some found,own⟩,rfl⟩
    obtain ⟨w,wmem,role,raw,got,eq⟩ := emitted_iff.mp (records.mem_iff.mp inLog)
    have same := eq.trans selectedFields
    have move := PvsBoardTree.play_injective mc fc f.position (congrArg ChildReturn.board same)
    have roleEq : role = pv := congrArg ChildReturn.pv same
    have result : raw = selected.score := congrArg ChildReturn.score same
    obtain rfl := at_visit unique wmem mem move
    subst role
    rw [returned] at got
    exact congrArg some (result.symm.trans (Option.some.inj got).symm)

end NeuralReversi.PvsParallel.FrameReturns.OwnerCalls
