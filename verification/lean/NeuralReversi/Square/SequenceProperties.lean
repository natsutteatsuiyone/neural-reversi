import NeuralReversi.Square.SequenceChunkProperties
namespace NeuralReversi.Square
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000

def PairEncodes (f r : Byte) (sq : Square) : Prop :=
  validFile f ∧ validRank r ∧ sq.val = 8 * (r.toNat - 49) + fileNumber f

inductive EncodesMoves : List Byte → List Square → Prop
  | nil : EncodesMoves [] []
  | cons {f r rest sq moves} : PairEncodes f r sq → EncodesMoves rest moves →
      EncodesMoves (f :: r :: rest) (sq :: moves)

inductive BadSequence : List Byte → Nat → ParseError → Prop
  | single {a reason} : ChunkErrorSpec [a] reason → BadSequence [a] 0 reason
  | here {f r rest reason} : ChunkErrorSpec [f,r] reason → BadSequence (f::r::rest) 0 reason
  | there {f r rest sq k reason} : PairEncodes f r sq → BadSequence rest k reason →
      BadSequence (f::r::rest) (k+1) reason

theorem parseChunks_cons_ok (index : BitVec 64) (bytes : List Byte) (rest : List (List Byte))
    (moves : List Square) :
    parseChunks index (bytes::rest) = .ok moves ↔
      ∃ sq tail, parseChunk bytes = .ok sq ∧ parseChunks (index+1#64) rest = .ok tail ∧ moves = sq::tail := by
  cases hc : parseChunk bytes with
  | ok sq => cases ht : parseChunks (index+1#64) rest <;> simp [parseChunks,hc,ht,Except.map,eq_comm]
  | error e => cases e <;> simp [parseChunks,hc]

theorem parseChunks_cons_error (index : BitVec 64) (bytes : List Byte) (rest : List (List Byte))
    (err : SeqError) :
    parseChunks index (bytes::rest) = .error (.error err) ↔
      (parseChunk bytes = .error (.error err.source) ∧ err.index = index+1#64) ∨
      ∃ sq, parseChunk bytes = .ok sq ∧ parseChunks (index+1#64) rest = .error (.error err) := by
  rcases err with ⟨pos,reason⟩
  cases hc : parseChunk bytes with
  | ok sq => cases ht : parseChunks (index+1#64) rest <;> simp [parseChunks,hc,ht,Except.map]
  | error e => cases e <;> simp [parseChunks,hc,eq_comm,SeqError.mk.injEq,and_comm]

theorem encodes_nil (moves : List Square) : EncodesMoves [] moves ↔ moves = [] := by
  constructor
  · intro h; cases h; rfl
  · rintro rfl; exact .nil

theorem encodes_single (a : Byte) (moves : List Square) : ¬ EncodesMoves [a] moves := by
  intro h; cases h

theorem encodes_cons (f r : Byte) (rest : List Byte) (moves : List Square) :
    EncodesMoves (f::r::rest) moves ↔
      ∃ sq tail, moves = sq::tail ∧ PairEncodes f r sq ∧ EncodesMoves rest tail := by
  constructor
  · intro h; cases h with | cons pair tail => exact ⟨_,_,rfl,pair,tail⟩
  · rintro ⟨sq,tail,rfl,pair,ht⟩; exact .cons pair ht

theorem parseChunks_raw_success (index : BitVec 64) (bytes : List Byte) (moves : List Square) :
    parseChunks index (chunks2 bytes) = .ok moves ↔ EncodesMoves bytes moves := by
  induction bytes using chunks2.induct generalizing index moves with
  | case1 => simp [chunks2,parseChunks,encodes_nil,eq_comm]
  | case2 a =>
    rw [chunks2,parseChunks_cons_ok]
    simp [parseChunk_success_iff [a] (by simp),encodes_single]
  | case3 f r rest ih =>
    rw [chunks2,parseChunks_cons_ok,encodes_cons]
    simp only [ih]
    constructor
    · rintro ⟨sq,tail,pair,ht,rfl⟩
      have parsed := (parseChunk_success_iff [f,r] (by simp) sq).mp pair
      obtain ⟨f2,r2,he,hf,hr,coord⟩ := parsed
      have ee := List.cons.inj he
      have er := List.cons.inj ee.2
      rcases ee with ⟨rfl,_⟩
      rcases er with ⟨rfl,_⟩
      exact ⟨sq,tail,rfl,⟨hf,hr,coord⟩,ht⟩
    · rintro ⟨sq,tail,rfl,pair,ht⟩
      exact ⟨sq,tail,parseChunk_success_reverse f r sq pair.1 pair.2.1 pair.2.2,ht,rfl⟩

theorem parseChunk_pair_iff (f r : Byte) (sq : Square) :
    parseChunk [f,r] = .ok sq ↔ PairEncodes f r sq := by
  constructor
  · intro h
    obtain ⟨f2,r2,eq,hf,hr,coord⟩ := (parseChunk_success_iff [f,r] (by simp) sq).mp h
    have ee := List.cons.inj eq
    rcases ee with ⟨rfl,ee⟩
    rcases List.cons.inj ee with ⟨rfl,_⟩
    exact ⟨hf,hr,coord⟩
  · intro h; exact parseChunk_success_reverse f r sq h.1 h.2.1 h.2.2

theorem index_shift (index : BitVec 64) (k : Nat) :
    (index+1#64)+BitVec.ofNat 64 (k+1) = index+BitVec.ofNat 64 (k+1+1) := by
  rw [BitVec.add_assoc]
  congr 1
  change BitVec.ofNat 64 1 + BitVec.ofNat 64 (k+1) = _
  rw [← BitVec.ofNat_add]
  congr 1
  omega

theorem parseChunks_raw_error (index : BitVec 64) (bytes : List Byte) (err : SeqError) :
    parseChunks index (chunks2 bytes) = .error (.error err) ↔
      ∃ k, BadSequence bytes k err.source ∧ err.index = index+BitVec.ofNat 64 (k+1) := by
  induction bytes using chunks2.induct generalizing index err with
  | case1 =>
    constructor
    · intro h; cases h
    · rintro ⟨k,h,_⟩; cases h
  | case2 a =>
    rw [chunks2,parseChunks_cons_error]
    simp only [parseChunks]
    constructor
    · rintro (⟨bad,idx⟩ | ⟨sq,_,impossible⟩)
      · exact ⟨0,.single ((parseChunk_error_iff [a] err.source).mp bad),idx⟩
      · cases impossible
    · rintro ⟨k,bad,idx⟩
      cases bad with
      | single he => exact .inl ⟨(parseChunk_error_iff [a] err.source).mpr he,idx⟩
  | case3 f r rest ih =>
    rw [chunks2,parseChunks_cons_error]
    constructor
    · rintro (⟨bad,idx⟩ | ⟨sq,good,bad⟩)
      · exact ⟨0,.here ((parseChunk_error_iff [f,r] err.source).mp bad),idx⟩
      · obtain ⟨k,bad,idx⟩ := (ih (index+1#64) err).mp bad
        exact ⟨k+1,.there ((parseChunk_pair_iff f r sq).mp good) bad,by rw [idx,index_shift]⟩
    · rintro ⟨k,bad,idx⟩
      cases bad with
      | here he => exact .inl ⟨(parseChunk_error_iff [f,r] err.source).mpr he,idx⟩
      | @there _ _ _ sq n _ pair bad =>
        refine .inr ⟨sq,(parseChunk_pair_iff f r sq).mpr pair,?_⟩
        apply (ih (index+1#64) err).mpr
        exact ⟨n,bad,by rw [index_shift]; exact idx⟩

theorem parseChunks_no_panic (index : BitVec 64) (chunks : List (List Byte)) (panic : Panic) :
    parseChunks index chunks ≠ .error (.panic panic) := by
  induction chunks generalizing index with
  | nil => simp [parseChunks]
  | cons bytes rest ih =>
    cases hc : parseChunk bytes with
    | error failure =>
      cases failure with
      | error reason => simp [parseChunks,hc]
      | panic reason => exact False.elim (parseChunk_no_panic bytes reason hc)
    | ok sq =>
      simp only [parseChunks,hc]
      cases ht : parseChunks (index+1#64) rest with
      | ok tail => simp [Except.map]
      | error failure =>
        simp only [Except.map]
        intro eq
        exact ih (index+1#64) (ht.trans eq)

theorem take_two_succ (f r : Byte) (rest : List Byte) (k : Nat) :
    (f::r::rest).take (2*(k+1)) = f::r::rest.take (2*k) := by
  have h : 2*(k+1) = 2*k+2 := by omega
  rw [h]
  rfl

theorem drop_two_succ (f r : Byte) (rest : List Byte) (k : Nat) :
    (f::r::rest).drop (2*(k+1)) = rest.drop (2*k) := by
  have h : 2*(k+1) = 2*k+2 := by omega
  rw [h]
  rfl

theorem badSequence_prefix_forward {bytes : List Byte} {k : Nat} {reason : ParseError}
    (bad : BadSequence bytes k reason) :
    ∃ moves, 2*k < bytes.length ∧ EncodesMoves (bytes.take (2*k)) moves ∧
      ChunkErrorSpec ((bytes.drop (2*k)).take 2) reason := by
  induction bad with
  | single spec => exact ⟨[],by simp,.nil,spec⟩
  | here spec => exact ⟨[],by simp,.nil,spec⟩
  | @there f r rest sq k reason pair bad ih =>
    obtain ⟨moves,bound,enc,spec⟩ := ih
    refine ⟨sq::moves,?_,?_,?_⟩
    · simp only [List.length_cons]; omega
    · rw [take_two_succ]
      exact .cons pair enc
    · rw [drop_two_succ]
      exact spec

theorem badSequence_prefix_reverse (bytes : List Byte) (k : Nat) (reason : ParseError)
    (prefixEvidence : ∃ moves, 2*k < bytes.length ∧ EncodesMoves (bytes.take (2*k)) moves ∧
      ChunkErrorSpec ((bytes.drop (2*k)).take 2) reason) : BadSequence bytes k reason := by
  induction k generalizing bytes with
  | zero =>
    obtain ⟨moves,bound,enc,spec⟩ := prefixEvidence
    cases bytes with
    | nil => simp at bound
    | cons f rest =>
      cases rest with
      | nil => exact .single spec
      | cons r tail => exact .here spec
  | succ k ih =>
    obtain ⟨moves,bound,enc,spec⟩ := prefixEvidence
    cases bytes with
    | nil => simp at bound
    | cons f rest =>
      cases rest with
      | nil => simp at bound
      | cons r tail =>
        rw [take_two_succ] at enc
        rw [drop_two_succ] at spec
        obtain ⟨sq,ms,_,pair,ht⟩ := (encodes_cons f r (tail.take (2*k)) moves).mp enc
        refine .there pair (ih tail ⟨ms,?_,ht,spec⟩)
        simp only [List.length_cons] at bound
        omega

theorem badSequence_prefix_iff (bytes : List Byte) (k : Nat) (reason : ParseError) :
    BadSequence bytes k reason ↔
      ∃ moves, 2*k < bytes.length ∧ EncodesMoves (bytes.take (2*k)) moves ∧
        ChunkErrorSpec ((bytes.drop (2*k)).take 2) reason :=
  ⟨badSequence_prefix_forward, badSequence_prefix_reverse bytes k reason⟩

theorem parseSequence_success_iff (text : String) (moves : List Square) :
    parseSequence text = .ok moves ↔ EncodesMoves (trimmedBytes text) moves :=
  parseChunks_raw_success 0#64 (trimmedBytes text) moves

theorem parseSequence_error_iff (text : String) (domain : SequenceDomain text) (err : SeqError) :
    parseSequence text = .error (.error err) ↔
      ∃ k moves, 2*k < (trimmedBytes text).length ∧
        EncodesMoves ((trimmedBytes text).take (2*k)) moves ∧
        ChunkErrorSpec (((trimmedBytes text).drop (2*k)).take 2) err.source ∧
        err.index.toNat = k+1 := by
  rw [parseSequence,parseChunks_raw_error]
  have lengthBound := sequenceDomain_trimmed text domain
  constructor
  · rintro ⟨k,bad,idx⟩
    obtain ⟨moves,bound,enc,spec⟩ := badSequence_prefix_forward bad
    refine ⟨k,moves,bound,enc,spec,?_⟩
    have kb : k+1 < 2^64 := by omega
    rw [idx,BitVec.zero_add,BitVec.toNat_ofNat,Nat.mod_eq_of_lt kb]
  · rintro ⟨k,moves,bound,enc,spec,idx⟩
    refine ⟨k,(badSequence_prefix_iff _ _ _).mpr ⟨moves,bound,enc,spec⟩,?_⟩
    apply BitVec.eq_of_toNat_eq
    have kb : k+1 < 2^64 := by omega
    simpa only [BitVec.zero_add,BitVec.toNat_ofNat,Nat.mod_eq_of_lt kb] using idx

theorem parseSequence_no_panic (text : String) (panic : Panic) :
    parseSequence text ≠ .error (.panic panic) :=
  parseChunks_no_panic 0#64 (chunks2 (trimmedBytes text)) panic

theorem EncodesMoves.length_eq {bytes : List Byte} {moves : List Square}
    (encoded : EncodesMoves bytes moves) : bytes.length = 2*moves.length := by
  induction encoded with
  | nil => rfl
  | cons pair tail ih => simp only [List.length_cons]; omega

theorem EncodesMoves.playable {bytes : List Byte} {moves : List Square}
    (encoded : EncodesMoves bytes moves) : ∀ sq ∈ moves, sq.val < 64 := by
  induction encoded with
  | nil => simp
  | @cons f r rest sq moves pair tail ih =>
    intro square member
    rcases List.mem_cons.mp member with eq | member
    · subst square
      obtain ⟨other,_,coord,lt⟩ := parse_pair_valid f r pair.1 pair.2.1
      have sqcoord := pair.2.2
      omega
    · exact ih square member

def SequenceContract : Prop :=
  ∀ text : String,
    (∀ moves, parseSequence text = .ok moves ↔ EncodesMoves (trimmedBytes text) moves) ∧
    (∀ moves, parseSequence text = .ok moves →
      (trimmedBytes text).length = 2*moves.length ∧ ∀ sq ∈ moves, sq.val < 64) ∧
    (∀ panic, parseSequence text ≠ .error (.panic panic)) ∧
    (SequenceDomain text → ∀ err,
      parseSequence text = .error (.error err) ↔
        ∃ k moves, 2*k < (trimmedBytes text).length ∧
          EncodesMoves ((trimmedBytes text).take (2*k)) moves ∧
          ChunkErrorSpec (((trimmedBytes text).drop (2*k)).take 2) err.source ∧
          err.index.toNat = k+1)

theorem sequenceContract_proven : SequenceContract := by
  intro text
  refine ⟨parseSequence_success_iff text,?_,parseSequence_no_panic text,parseSequence_error_iff text⟩
  intro moves success
  have encoded := (parseSequence_success_iff text moves).mp success
  exact ⟨encoded.length_eq,encoded.playable⟩

theorem encodes_empty_iff (bytes : List Byte) : EncodesMoves bytes [] ↔ bytes = [] := by
  constructor
  · intro encoded; cases encoded; rfl
  · rintro rfl; exact .nil

theorem parseSequence_empty_iff (text : String) :
    parseSequence text = .ok [] ↔ trimmedBytes text = [] := by
  rw [parseSequence_success_iff, encodes_empty_iff]

theorem parseSequence_trimmed (text : String) :
    parseSequence (String.ofList (trimWhitespace text.toList)) = parseSequence text := by
  simp only [parseSequence, trimmedBytes, String.toList_ofList, trimWhitespace_idempotent]

end NeuralReversi.Square
