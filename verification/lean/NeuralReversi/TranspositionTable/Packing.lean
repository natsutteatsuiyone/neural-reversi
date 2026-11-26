import NeuralReversi.TranspositionTable.Proofs
import NeuralReversi.TranspositionTable.SeqLock

namespace NeuralReversi.TranspositionTable.Packing

def pack (bigEndian : Bool) (e : EntryData) : BitVec 64 :=
  BitVec.ofNat 64 (if bigEndian then
    e.score.toNat * 281474976710656 + e.bestMove.toNat * 1099511627776 +
    e.bound.toNat * 4294967296 + e.depth.toNat * 16777216 +
    e.selectivity.toNat * 65536 + e.generation.toNat * 256 + e.flags.toNat
  else
    e.score.toNat + e.bestMove.toNat * 65536 + e.bound.toNat * 16777216 +
    e.depth.toNat * 4294967296 + e.selectivity.toNat * 1099511627776 +
    e.generation.toNat * 281474976710656 + e.flags.toNat * 72057594037927936)

def unpack (bigEndian : Bool) (raw : BitVec 64) : EntryData :=
  if bigEndian then
    ⟨BitVec.ofNat 16 (raw.toNat / 281474976710656),
     BitVec.ofNat 8 (raw.toNat / 1099511627776),
     BitVec.ofNat 8 (raw.toNat / 4294967296),
     BitVec.ofNat 8 (raw.toNat / 16777216),
     BitVec.ofNat 8 (raw.toNat / 65536),
     BitVec.ofNat 8 (raw.toNat / 256), BitVec.ofNat 8 raw.toNat⟩
  else
    ⟨BitVec.ofNat 16 raw.toNat, BitVec.ofNat 8 (raw.toNat / 65536),
     BitVec.ofNat 8 (raw.toNat / 16777216),
     BitVec.ofNat 8 (raw.toNat / 4294967296),
     BitVec.ofNat 8 (raw.toNat / 1099511627776),
     BitVec.ofNat 8 (raw.toNat / 281474976710656),
     BitVec.ofNat 8 (raw.toNat / 72057594037927936)⟩

theorem unpack_pack (bigEndian : Bool) (e : EntryData) :
    unpack bigEndian (pack bigEndian e) = e := by
  have hs : e.score.toNat < 65536 := e.score.toFin.isLt
  have hm : e.bestMove.toNat < 256 := e.bestMove.toFin.isLt
  have hb : e.bound.toNat < 256 := e.bound.toFin.isLt
  have hd : e.depth.toNat < 256 := e.depth.toFin.isLt
  have hl : e.selectivity.toNat < 256 := e.selectivity.toFin.isLt
  have hg : e.generation.toNat < 256 := e.generation.toFin.isLt
  have hf : e.flags.toNat < 256 := e.flags.toFin.isLt
  cases e
  cases bigEndian <;> dsimp [pack, unpack]
  all_goals
    congr 1 <;> apply BitVec.eq_of_toNat_eq <;>
      simp only [BitVec.toNat_ofNat] <;> omega

theorem pack_unpack (bigEndian : Bool) (raw : BitVec 64) :
    pack bigEndian (unpack bigEndian raw) = raw := by
  have hr : raw.toNat < 18446744073709551616 := raw.toFin.isLt
  cases bigEndian <;> apply BitVec.eq_of_toNat_eq
  all_goals
    dsimp [pack, unpack]
    simp only [BitVec.toNat_ofNat]
    omega

def LayoutAt (bigEndian : Bool) (raw : BitVec 64) (e : EntryData) : Prop :=
  let n := raw.toNat
  if bigEndian then
    n / 281474976710656 % 65536 = e.score.toNat ∧
    n / 1099511627776 % 256 = e.bestMove.toNat ∧
    n / 4294967296 % 256 = e.bound.toNat ∧
    n / 16777216 % 256 = e.depth.toNat ∧
    n / 65536 % 256 = e.selectivity.toNat ∧
    n / 256 % 256 = e.generation.toNat ∧ n % 256 = e.flags.toNat
  else
    n % 65536 = e.score.toNat ∧ n / 65536 % 256 = e.bestMove.toNat ∧
    n / 16777216 % 256 = e.bound.toNat ∧
    n / 4294967296 % 256 = e.depth.toNat ∧
    n / 1099511627776 % 256 = e.selectivity.toNat ∧
    n / 281474976710656 % 256 = e.generation.toNat ∧
    n / 72057594037927936 % 256 = e.flags.toNat

theorem unpack_layout (bigEndian : Bool) (raw : BitVec 64) :
    LayoutAt bigEndian raw (unpack bigEndian raw) := by
  cases bigEndian <;> simp [LayoutAt, unpack, BitVec.toNat_ofNat]

theorem pack_layout (bigEndian : Bool) (e : EntryData) :
    LayoutAt bigEndian (pack bigEndian e) e := by
  have h := unpack_layout bigEndian (pack bigEndian e)
  rw [unpack_pack] at h
  exact h

theorem pack_injective (bigEndian : Bool) (a b : EntryData)
    (same : pack bigEndian a = pack bigEndian b) : a = b := by
  have h := congrArg (unpack bigEndian) same
  simpa only [unpack_pack] using h

def noMove : BitVec 8 := 255#8

def encodeBestMove : Option (Fin 64) → BitVec 8
  | some sq => BitVec.ofNat 8 sq.val
  | none => noMove

def decodeBestMove (raw : BitVec 8) : Option (Fin 64) :=
  if h : raw.toNat < 64 then some ⟨raw.toNat, h⟩ else none

theorem decodeBestMove_encodeBestMove (move : Option (Fin 64)) :
    decodeBestMove (encodeBestMove move) = move := by
  cases move with
  | none => decide
  | some sq =>
    have hsq : sq.val < 64 := sq.isLt
    have hmod : sq.val % 256 = sq.val := Nat.mod_eq_of_lt (by omega)
    simp [encodeBestMove, decodeBestMove, BitVec.toNat_ofNat, hmod, hsq]

theorem decodeBestMove_noMove : decodeBestMove noMove = none := by
  decide

def EnumSafe (e : EntryData) : Prop :=
  (e.bestMove.toNat < 64 ∨ e.bestMove = noMove) ∧ e.bound.toNat < 4

def boundByte : Bound → BitVec 8
  | .none => 0 | .lower => 1 | .upper => 2 | .exact => 3

def newData (score : Score) (move : Option (Fin 64)) (bound : Bound) (depth : Depth)
    (selectivity : Selectivity) (generation : BitVec 8) (endgame pv : Bool) : EntryData :=
  ⟨score.raw.setWidth 16, encodeBestMove move, boundByte bound,
    depth.setWidth 8, BitVec.ofNat 8 selectivity.val, generation &&& 127,
    (if endgame then 1 else 0) ||| (if pv then 2 else 0)⟩

def withBestMove (e : EntryData) (move : Option (Fin 64)) : EntryData :=
  { e with bestMove := encodeBestMove move }

def withBestMoveOf (e other : EntryData) : EntryData :=
  { e with bestMove := other.bestMove }

def penalized (e : EntryData) : EntryData := { e with depth := e.depth - 1 }

def zeroData : EntryData := ⟨0, 0, 0, 0, 0, 0, 0⟩

theorem encodeBestMove_safe (move : Option (Fin 64)) :
    (encodeBestMove move).toNat < 64 ∨ encodeBestMove move = noMove := by
  cases move with
  | none => exact Or.inr rfl
  | some sq =>
    have hsq := sq.isLt
    left
    simp only [encodeBestMove, BitVec.toNat_ofNat]
    omega

theorem newData_enumSafe (score : Score) (move : Option (Fin 64)) (bound : Bound) (depth : Depth)
    (selectivity : Selectivity) (generation : BitVec 8) (endgame pv : Bool) :
    EnumSafe (newData score move bound depth selectivity generation endgame pv) := by
  constructor
  · exact encodeBestMove_safe move
  · cases bound <;> simp [newData, boundByte]

theorem withBestMove_enumSafe (e : EntryData) (move : Option (Fin 64)) (safe : EnumSafe e) :
    EnumSafe (withBestMove e move) :=
  ⟨encodeBestMove_safe move, safe.2⟩

theorem withBestMoveOf_enumSafe (e other : EntryData) (safe : EnumSafe e)
    (safeOther : EnumSafe other) : EnumSafe (withBestMoveOf e other) :=
  ⟨safeOther.1, safe.2⟩

theorem penalized_enumSafe (e : EntryData) (safe : EnumSafe e) :
    EnumSafe (penalized e) := safe

theorem zeroData_enumSafe : EnumSafe zeroData := by
  unfold EnumSafe
  decide

inductive Produced : EntryData → Prop where
  | zero : Produced zeroData
  | fresh (score : Score) (move : Option (Fin 64)) (bound : Bound) (depth : Depth)
      (selectivity : Selectivity) (generation : BitVec 8) (endgame pv : Bool) :
      Produced (newData score move bound depth selectivity generation endgame pv)
  | keepMove {e : EntryData} (prior : Produced e) (move : Option (Fin 64)) :
      Produced (withBestMove e move)
  | keepMoveOf {e other : EntryData} (prior : Produced e) (priorOther : Produced other) :
      Produced (withBestMoveOf e other)
  | penalize {e : EntryData} (prior : Produced e) (positive : 0 < e.depth.toNat) :
      Produced (penalized e)

theorem produced_enumSafe {e : EntryData} (produced : Produced e) : EnumSafe e := by
  induction produced with
  | zero => exact zeroData_enumSafe
  | fresh score move bound depth selectivity generation endgame pv =>
      exact newData_enumSafe score move bound depth selectivity generation endgame pv
  | @keepMove e _ move ih => exact withBestMove_enumSafe e move ih
  | @keepMoveOf e other _ _ ih ihOther => exact withBestMoveOf_enumSafe e other ih ihOther
  | @penalize e _ _ ih => exact penalized_enumSafe e ih

def RepresentationContract : Prop :=
  (∀ bigEndian raw, pack bigEndian (unpack bigEndian raw) = raw ∧
    LayoutAt bigEndian raw (unpack bigEndian raw)) ∧
  (∀ bigEndian entry, unpack bigEndian (pack bigEndian entry) = entry ∧
    LayoutAt bigEndian (pack bigEndian entry) entry)

theorem representationContract_proven : RepresentationContract :=
  ⟨fun endian raw => ⟨pack_unpack endian raw, unpack_layout endian raw⟩,
    fun endian entry => ⟨unpack_pack endian entry, pack_layout endian entry⟩⟩

def DiscriminantContract : Prop :=
  (∀ move, decodeBestMove (encodeBestMove move) = move) ∧
  (∀ bigEndian entry, Produced entry → EnumSafe (unpack bigEndian (pack bigEndian entry)))

theorem discriminantContract_proven : DiscriminantContract := by
  refine ⟨decodeBestMove_encodeBestMove, ?_⟩
  intro endian entry produced
  rw [unpack_pack]
  exact produced_enumSafe produced

theorem withBestMove_canCut (e : EntryData) (move : Option (Fin 64)) (checked : Bool)
    (beta : Score) (depth : Depth) (selectivity : Selectivity) (requireEndgame : Bool) :
    canCut checked (withBestMove e move) beta depth selectivity requireEndgame =
      canCut checked e beta depth selectivity requireEndgame := by rfl

theorem withBestMoveOf_canCut (e other : EntryData) (checked : Bool)
    (beta : Score) (depth : Depth) (selectivity : Selectivity) (requireEndgame : Bool) :
    canCut checked (withBestMoveOf e other) beta depth selectivity requireEndgame =
      canCut checked e beta depth selectivity requireEndgame := by rfl

theorem roundtrip_canCut (bigEndian checked : Bool) (e : EntryData)
    (beta : Score) (depth : Depth) (selectivity : Selectivity) (requireEndgame : Bool) :
    canCut checked (unpack bigEndian (pack bigEndian e)) beta depth selectivity requireEndgame =
      canCut checked e beta depth selectivity requireEndgame := by rw [unpack_pack]

theorem decode_published (bigEndian : Bool) (raw : BitVec 64) (e : EntryData)
    (stored : raw = pack bigEndian e) : unpack bigEndian raw = e := by
  rw [stored, unpack_pack]

theorem lookup_decodes_produced {Event : Type} {memory : SeqLock.Memory Event}
    {epoch finalIndex : Nat} (writers : SeqLock.Writers memory)
    (reader : SeqLock.Reader memory epoch finalIndex)
    {pe oe de : Nat} (pr : SeqLock.FieldRead reader .player pe)
    (op : SeqLock.FieldRead reader .opponent oe) (dr : SeqLock.FieldRead reader .metadata de)
    (qp qo result : SeqLock.Word) (bigEndian : Bool) (e : EntryData)
    (produced : Produced e)
    (encoded : memory.writeValue (.payload .metadata) epoch = pack bigEndian e)
    (noWrap : finalIndex < 2 * epoch + SeqLock.modulus)
    (accepted : SeqLock.lookupDecision (memory.readValue reader.first)
      (memory.readValue reader.last)
      ⟨memory.readValue pr.event, memory.readValue op.event, memory.readValue dr.event⟩ qp qo =
        some result) :
    memory.writeValue (.payload .player) epoch = qp ∧
    memory.writeValue (.payload .opponent) epoch = qo ∧
    unpack bigEndian result = e ∧ EnumSafe (unpack bigEndian result) := by
  obtain ⟨hp, ho, hd⟩ := SeqLock.lookup_consistent writers reader pr op dr qp qo result
    noWrap accepted
  have decoded := decode_published bigEndian result e (hd.trans encoded)
  exact ⟨hp, ho, decoded, decoded.symm ▸ produced_enumSafe produced⟩

theorem decoded_cutoff_preserved (bigEndian checked : Bool) (raw : BitVec 64) (e : EntryData)
    (stored : raw = pack bigEndian e) (beta : Score) (depth : Depth)
    (selectivity : Selectivity) (requireEndgame : Bool) :
    canCut checked (unpack bigEndian raw) beta depth selectivity requireEndgame =
      canCut checked e beta depth selectivity requireEndgame := by
  rw [decode_published bigEndian raw e stored]

private def layoutFixture : EntryData := ⟨0x2211, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88⟩

example : pack false layoutFixture = 0x8877665544332211 := by rfl
example : pack true layoutFixture = 0x2211334455667788 := by rfl
example : unpack false 0x8877665544332211 = layoutFixture := by rfl
example : unpack true 0x2211334455667788 = layoutFixture := by rfl
example : unpack false 0 = zeroData := by rfl
example : unpack true 0 = zeroData := by rfl

example : ¬ EnumSafe (unpack false 0x0000000004ff0000) := by
  unfold EnumSafe
  decide
example : ¬ EnumSafe (unpack true 0x0000ff0400000000) := by
  unfold EnumSafe
  decide
example : ¬ EnumSafe (unpack false 0x0000000003400000) := by
  unfold EnumSafe
  decide
example : EnumSafe (unpack false 0x0000000003ff0000) := by
  unfold EnumSafe
  decide

example : (newData ⟨32768⟩ none .exact 256 ⟨5, by decide⟩ 255 true true).score.toInt =
    -32768 := by rfl
example : (newData ⟨0⟩ none .exact 256 ⟨5, by decide⟩ 255 true true).depth = 0 := by rfl
example : (newData ⟨0⟩ none .exact 0 ⟨5, by decide⟩ 255 true true).generation = 127 := by rfl
example : (newData ⟨0⟩ none .exact 0 ⟨5, by decide⟩ 255 true true).flags = 3 := by rfl
example : (newData ⟨0⟩ none .exact 0 ⟨5, by decide⟩ 255 true true).bestMove = noMove := by rfl

end NeuralReversi.TranspositionTable.Packing
