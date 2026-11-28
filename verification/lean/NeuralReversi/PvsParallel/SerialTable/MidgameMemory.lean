module
public import NeuralReversi.AspirationMir.SerialTable.Repeated
public import NeuralReversi.PvsParallel.SerialTable.Fallback

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialTable.MidgameMemory

open AspirationMir.SerialTable.Repeated (ReadyMemory)
open AspirationMir.SerialTable.PvInterior (InsideRange)
open EndgameSearch.CacheHistory (Pool)
open PvsBoardTree (Position GeneratedSquare)
open TranspositionTable
open SerialCalls (Request)
open SerialCalls.ScoreWords (Finite)
open PvsMoveLoop (scoreMax)

/-- The caller keeps the entry slot while descendants can replace its contents. -/
def save (m : Memory l) (slot : l.Index) (b : Position) (v : Int) (tag : Bound)
    (d : Depth) (hint : Option (GeneratedSquare mc b.val)) (level : Selectivity) (ttPv : Bool) : Memory l :=
  ⟨m.tt.store slot b (Writes.word v) tag d (hint.map Subtype.val) level false ttPv,
    m.tt.store_supported m.supported ..,m.cache⟩

theorem save_ready (m : Memory l) (ready : ReadyMemory p mc fc m) (slot : l.Index) (b : Position)
    (v : Int) (tag : Bound) (d : Depth) (hint : Option (GeneratedSquare mc b.val))
    (level : Selectivity) (ttPv : Bool)
    (exactScore : tag = .exact → -scoreMax ≤ v ∧ v ≤ scoreMax) :
    ReadyMemory p mc fc (save m slot b v tag d hint level ttPv) := by
  have legal : TableReads.LegalChoice b (hint.map Subtype.val) := by
    intro sq h
    obtain ⟨move,_,rfl⟩ := Option.map_eq_some_iff.mp h
    exact (Board.getMoves_exact mc b.val _).mp move.property
  refine ⟨⟨⟨m.tt.store_other ready.1.1.1 _ _ _ _ _ _ _ false _ (Or.inl rfl) legal,ready.1.1.2⟩,
    m.tt.store_finite m.supported ready.1.2 _ _ _ _ _ _ _ _ _⟩,m.tt.store_exactScores m.supported ready.2 _ _ _ _ _ _ _ _ _ ?_⟩
  intro tag
  have range := exactScore tag
  have eq : (Writes.word v).raw.toInt = v := BitVec.toInt_ofInt_eq_self (by decide)
    (by unfold scoreMax at range; omega) (by unfold scoreMax at range; omega)
  rwa [eq]

theorem classified (q : Request) (v : Int) (win : Aspiration.Window q.alpha q.beta)
    (finite : Finite v) (range : WeakTrial.Range q v) :
    classify q.pv (Writes.word v) (Writes.word q.alpha) (Writes.word q.beta) = .exact →
      -scoreMax ≤ v ∧ v ≤ scoreMax := by
  intro tag
  have h := ((classify_contract ..).2.1).mp tag
  obtain ⟨lo,mid,hi⟩ := win
  obtain ⟨vl,vh⟩ := finite
  have repr {x : Int} (low : -32767 ≤ x) (high : x ≤ 32767) : (Writes.word x).raw.toInt = x :=
    BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
  rw [repr (by omega : -32767 ≤ v) (by omega),repr lo (by omega),repr (by omega) hi] at h
  exact range h.1 h.2.1 h.2.2

/-- The serial penalty rechecks the same slot and expected payload before publishing its depth decrement. -/
def penalize (m : Memory l) (b : Position) : Memory l :=
  let slot := m.tt.image.slot b
  let old := m.tt.image.get slot
  match m.tt.image.lookup b with
  | none => m
  | some e =>
    let seen := Snapshot.read l.bigEndian old.seq old.seq (payload old)
    match Penalty.tryPenalize l.bigEndian b.val.player b.val.opponent e seen old.seq with
    | none => m
    | some v => ⟨m.tt.put slot b (SeqLock.publish old.seq) v,m.tt.put_supported slot b v m.supported,m.cache⟩

theorem penalty_ready (m : Memory l) (b : Position) (ready : ReadyMemory p mc fc m) :
    ReadyMemory p mc fc (penalize m b) := by
  dsimp only [penalize]
  split <;> try exact ready
  next e read =>
    split <;> try exact ready
    next v made =>
      obtain ⟨positive,_,_,_,_,_,_,rfl⟩ := (Penalty.writeContract_proven ..).mp made
      obtain ⟨hit⟩ := m.tt.image.lookup_source m.supported b read
      have source := hit.source
      have extend {P : Position → EntryData → Prop} (old : ∀ x y, (x,y) ∈ m.tt.history → P x y)
          (fresh : P b (Packing.penalized e)) :
          ∀ x y, (x,y) ∈ m.tt.history ++ [(b,Packing.penalized e)] → P x y := by
        intro x y mem
        rcases List.mem_append.mp mem with prior | last
        · exact old _ _ prior
        · cases List.mem_singleton.mp last; exact fresh
      refine ⟨⟨⟨⟨m.tt.put_supported _ _ _ m.supported, .append ready.1.1.1.publications
        (.penalized b e source positive),extend ready.1.1.1.hints (ready.1.1.1.hints b e source)⟩,ready.1.1.2⟩,
        extend (P := fun _ e => Finite e.score.toInt) ready.1.2 (ready.1.2 b e source)⟩,
        extend (P := fun _ e => e.bound = Packing.boundByte .exact → _) ready.2 (ready.2 b e source)⟩

end NeuralReversi.PvsParallel.SerialTable.MidgameMemory
