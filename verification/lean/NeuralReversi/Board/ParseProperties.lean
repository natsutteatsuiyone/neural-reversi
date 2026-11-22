import NeuralReversi.Board.ParseModel
import NeuralReversi.Bitboard.SquareProperties
namespace NeuralReversi.Board
open Bitboard
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000

def validCell (c : Char) : Bool := c == 'X' || c == 'O' || c == '-'
def firstInvalid (tokens : List (AnySquare × Char)) := tokens.find? fun t => !validCell t.2

def bitsFor (c : Char) (tokens : List (AnySquare × Char)) : Bits :=
  tokens.foldr (fun t bits => if t.2 = c then ((1#64 <<< t.1.val) ||| bits) else bits) 0

def parsedBoard (side : Disc.PlayerColor) (initial : Board) (tokens : List (AnySquare × Char)) : Board :=
  ⟨initial.player ||| bitsFor side.toChar tokens,
   initial.opponent ||| bitsFor side.opposite.toChar tokens⟩

theorem bitsFor_nil (c : Char) : bitsFor c [] = 0 := by
  rfl

theorem bitsFor_cons (c ch : Char) (sq : AnySquare) (rest : List (AnySquare × Char)) :
    bitsFor c ((sq,ch)::rest) =
      (if ch = c then squareMask sq else 0) ||| bitsFor c rest := by
  by_cases hc : ch = c <;> simp [bitsFor, squareMask, hc]

theorem setSquare_playable (checked : Bool) (bits : Bits) (sq : AnySquare) :
    setSquare checked checked bits sq = .ok (bits ||| squareMask sq) := by
  have hm := (squareMaskContract_proven sq checked checked).2.1
  rw [setSquare, hm]
  rfl

theorem parseTokens_exact (checked : Bool) (side : Disc.PlayerColor)
    (initial : Board) (tokens : List (AnySquare × Char)) :
    parseTokens checked side initial tokens =
      match firstInvalid tokens with
      | some (sq,c) => .error (.error (.invalidChar c (BitVec.ofNat 64 sq.val)))
      | none => .ok (parsedBoard side initial tokens) := by
  induction tokens generalizing initial with
  | nil => simp [parseTokens, firstInvalid, parsedBoard, bitsFor_nil]
  | cons t rest ih =>
    rcases t with ⟨sq,c⟩
    have distinct : side.toChar ≠ side.opposite.toChar := by
      cases side <;> decide
    have v1 : validCell side.toChar = true := by cases side <;> decide
    have v2 : validCell side.opposite.toChar = true := by cases side <;> decide
    by_cases hp : c = side.toChar
    · subst c
      simp only [parseTokens, setSquare_playable checked initial.player sq]
      rw [ih]
      simp only [firstInvalid, List.find?_cons, v1, Bool.not_true, ↓reduceIte]
      cases hf : firstInvalid rest <;> simp_all [firstInvalid, parsedBoard, bitsFor_cons, BitVec.or_assoc]
    · by_cases ho : c = side.opposite.toChar
      · subst c
        simp only [parseTokens, ite_eq_right hp, setSquare_playable checked initial.opponent sq]
        rw [ih]
        simp only [firstInvalid, List.find?_cons, v2, Bool.not_true, ↓reduceIte]
        cases hf : firstInvalid rest <;> simp_all [firstInvalid, parsedBoard, bitsFor_cons, BitVec.or_assoc]
      · by_cases he : c = '-'
        · subst c
          simp only [parseTokens, ite_eq_right hp, ite_eq_right ho, ne_eq, not_true_eq_false, ↓reduceIte]
          rw [ih]
          simp [firstInvalid, validCell, parsedBoard, bitsFor_cons, hp, ho]
        · have hv : validCell c = false := by
            cases side <;> simp_all [validCell, Disc.PlayerColor.toChar, Disc.PlayerColor.opposite,
              Disc.PlayerColor.toDisc, Disc.toChar]
          simp [parseTokens, hp, ho, he, firstInvalid, hv]

theorem bitsFor_bit (c : Char) (tokens : List (AnySquare × Char)) (i : Nat) :
    (bitsFor c tokens).getLsbD i =
      tokens.any (fun t => decide (t.1.val = i ∧ t.2 = c)) := by
  induction tokens with
  | nil => simp [bitsFor_nil]
  | cons t rest ih =>
    rcases t with ⟨sq,ch⟩
    rw [bitsFor_cons]
    simp only [BitVec.getLsbD_or, ih, List.any_cons]
    by_cases hc : ch = c <;> simp [hc, squareMask_bit, eq_comm]

theorem fromString_exact (checked : Bool) (text : String) (side : Disc.PlayerColor)
    (representable : RepresentableText text) :
    fromString checked text side =
      if text.toList.length < 64 then
        .error (.error (.tooShort 64 (BitVec.ofNat 64 text.toList.length)))
      else if 64 < text.toList.length then
        .error (.error (.tooLong 64 (BitVec.ofNat 64 text.toList.length)))
      else match firstInvalid (Square.iter.zip text.toList) with
        | some (sq,c) => .error (.error (.invalidChar c (BitVec.ofNat 64 sq.val)))
        | none => .ok (parsedBoard side ⟨0,0⟩ (Square.iter.zip text.toList)) := by
  have count : (BitVec.ofNat 64 text.toList.length).toNat = text.toList.length := by
    simp only [BitVec.toNat_ofNat, Nat.mod_eq_of_lt representable]
  simp only [fromString, count]
  split
  · rfl
  · split
    · rfl
    · exact parseTokens_exact checked side _ _

theorem zip_iter_eq_ofFn (chars : List Char) (hlen : chars.length = 64) :
    Square.iter.zip chars = List.ofFn (fun i : Fin 64 =>
      ((i : AnySquare), chars[i.val]'(by omega))) := by
  apply List.ext_getElem
  · simp [Square.iter, hlen]
  · intro i hi hi2
    simp only [List.getElem_zip, Square.iter, List.getElem_finRange, List.getElem_ofFn]
    rfl

theorem bitsFor_zip_bit (chars : List Char) (hlen : chars.length = 64) (c : Char) (i : Fin 64) :
    (bitsFor c (Square.iter.zip chars)).getLsbD i.val = decide (chars[i.val]'(by omega) = c) := by
  rw [bitsFor_bit, zip_iter_eq_ofFn chars hlen]
  apply Bool.eq_iff_iff.mpr
  simp only [List.any_eq_true, List.mem_ofFn, decide_eq_true_eq]
  constructor
  · rintro ⟨t, ⟨j, rfl⟩, hj, hc⟩
    change j.val = i.val at hj
    have jeq : j = i := Fin.ext hj
    subst j
    exact hc
  · intro hc
    exact ⟨_, ⟨i,rfl⟩, rfl,hc⟩

theorem firstInvalid_none_iff (chars : List Char) (hlen : chars.length = 64) :
    firstInvalid (Square.iter.zip chars) = none ↔ ∀ c ∈ chars, validCell c = true := by
  rw [firstInvalid, List.find?_eq_none, zip_iter_eq_ofFn chars hlen]
  simp only [List.mem_ofFn, Bool.not_eq_true]
  constructor
  · intro h c hc
    obtain ⟨i, hi, eq⟩ := List.getElem_of_mem hc
    have got := h ((⟨i,by omega⟩ : AnySquare),c) ⟨⟨i,by omega⟩,by simp [eq]⟩
    simpa using got
  · intro h t ht
    obtain ⟨i,rfl⟩ := ht
    simpa using h _ (List.getElem_mem (n := i.val) (l := chars) (by omega))

theorem fromString_success_iff (checked : Bool) (text : String) (side : Disc.PlayerColor)
    (representable : RepresentableText text) :
    (∃ result, fromString checked text side = .ok result) ↔
      text.toList.length = 64 ∧ ∀ c ∈ text.toList, validCell c = true := by
  rw [fromString_exact checked text side representable]
  by_cases short : text.toList.length < 64
  · have ne : text.toList.length ≠ 64 := by omega
    simp [short, ne]
  · by_cases long : 64 < text.toList.length
    · have ne : text.toList.length ≠ 64 := by omega
      simp [short, long, ne]
    · have len : text.toList.length = 64 := by omega
      simp only [len, true_and]
      rw [← firstInvalid_none_iff text.toList len]
      cases firstInvalid (Square.iter.zip text.toList) <;> simp

theorem fromString_success_bits (checked : Bool) (text : String) (side : Disc.PlayerColor)
    (representable : RepresentableText text) (result : Board)
    (success : fromString checked text side = .ok result) :
    ∃ len : text.toList.length = 64, ∀ i : Fin 64,
      result.player.getLsbD i.val = decide (text.toList[i.val]'(by omega) = side.toChar) ∧
      result.opponent.getLsbD i.val = decide (text.toList[i.val]'(by omega) = side.opposite.toChar) := by
  obtain ⟨len,good⟩ := (fromString_success_iff checked text side representable).mp ⟨result,success⟩
  have findNone := (firstInvalid_none_iff text.toList len).mpr good
  rw [fromString_exact checked text side representable] at success
  simp [len, findNone] at success
  subst result
  refine ⟨len, ?_⟩
  intro i
  simp only [parsedBoard, BitVec.getLsbD_or, BitVec.getLsbD_zero, Bool.false_or,
    bitsFor_zip_bit _ len]
  trivial

theorem firstInvalid_some_iff (chars : List Char) (len : chars.length = 64)
    (sq : AnySquare) (c : Char) :
    firstInvalid (Square.iter.zip chars) = some (sq,c) ↔
      ∃ (i : Fin 64), sq.val = i.val ∧ chars[i.val]'(by omega) = c ∧ validCell c = false ∧
        ∀ (j : Fin 64), j.val < i.val → validCell (chars[j.val]'(by omega)) = true := by
  rw [firstInvalid, zip_iter_eq_ofFn chars len, List.find?_eq_some_iff_getElem]
  simp only [List.length_ofFn, List.getElem_ofFn, Bool.not_not]
  constructor
  · rintro ⟨invalid,i,hi,eq,earlier⟩
    have sqeq : sq.val = i := (congrArg (fun t : AnySquare × Char => t.1.val) eq).symm
    have ceq : chars[i]'(by omega) = c := congrArg Prod.snd eq
    refine ⟨⟨i,hi⟩,sqeq,ceq,?_,?_⟩
    · simpa using invalid
    · intro j hj
      simpa using earlier j.val hj
  · rintro ⟨i,sqeq,ceq,invalid,earlier⟩
    refine ⟨by simpa using invalid,i.val,i.isLt,?_,?_⟩
    · apply Prod.ext
      · exact Fin.ext sqeq.symm
      · exact ceq
    · intro j hj
      exact earlier ⟨j,by omega⟩ hj

theorem fromString_disjoint (checked : Bool) (text : String) (side : Disc.PlayerColor)
    (representable : RepresentableText text) (result : Board)
    (success : fromString checked text side = .ok result) : wellFormed result := by
  obtain ⟨len,bits⟩ := fromString_success_bits checked text side representable result success
  have distinct : side.toChar ≠ side.opposite.toChar := by cases side <;> decide
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  rw [BitVec.getLsbD_and, (bits ⟨i,hi⟩).1, (bits ⟨i,hi⟩).2]
  by_cases hp : text.toList[i]'(by omega) = side.toChar
  · simp [hp,distinct]
  · simp [hp]

def textErrorSpec (text : String) : TextError → Prop
  | .tooShort expected actual => text.toList.length < 64 ∧
      expected = 64 ∧ actual = BitVec.ofNat 64 text.toList.length
  | .tooLong expected actual => 64 < text.toList.length ∧
      expected = 64 ∧ actual = BitVec.ofNat 64 text.toList.length
  | .invalidChar c position => text.toList.length = 64 ∧
      ∃ sq, position = BitVec.ofNat 64 sq.val ∧ firstInvalid (Square.iter.zip text.toList) = some (sq,c)

theorem fromString_error_iff (checked : Bool) (text : String) (side : Disc.PlayerColor)
    (representable : RepresentableText text) (reason : TextError) :
    fromString checked text side = .error (.error reason) ↔ textErrorSpec text reason := by
  rw [fromString_exact checked text side representable]
  by_cases short : text.toList.length < 64
  · have notLong : ¬64 < text.toList.length := by omega
    have notEqual : text.toList.length ≠ 64 := by omega
    cases reason <;> simp [short,notLong,notEqual,textErrorSpec,eq_comm]
  · by_cases long : 64 < text.toList.length
    · have notEqual : text.toList.length ≠ 64 := by omega
      cases reason <;> simp [short,long,notEqual,textErrorSpec,eq_comm]
    · have len : text.toList.length = 64 := by omega
      cases findEq : firstInvalid (Square.iter.zip text.toList) with
      | none => cases reason <;> simp [len,findEq,textErrorSpec]
      | some pair =>
        rcases pair with ⟨sq,c⟩
        cases reason <;> simp [len,findEq,textErrorSpec,eq_comm]
        constructor
        · rintro ⟨hc,hp⟩
          exact ⟨sq,hp,rfl,hc⟩
        · rintro ⟨sq2,hp,rfl,hc⟩
          exact ⟨hc,hp⟩

theorem fromString_no_panic (checked : Bool) (text : String) (side : Disc.PlayerColor)
    (representable : RepresentableText text) (message : String) :
    fromString checked text side ≠ .error (.panic message) := by
  rw [fromString_exact checked text side representable]
  split
  · simp
  · split
    · simp
    · cases firstInvalid (Square.iter.zip text.toList) <;> simp

theorem validCell_iff (c : Char) : validCell c = true ↔ c = 'X' ∨ c = 'O' ∨ c = '-' := by
  simp only [validCell, Bool.or_eq_true, beq_iff_eq]
  simp only [or_assoc]

theorem fromString_invalidChar_iff (checked : Bool) (text : String) (side : Disc.PlayerColor)
    (representable : RepresentableText text) (c : Char) (position : Size) :
    fromString checked text side = .error (.error (.invalidChar c position)) ↔
      ∃ len : text.toList.length = 64,
        ∃ (i : Fin 64), position = BitVec.ofNat 64 i.val ∧ text.toList[i.val]'(by omega) = c ∧
          validCell c = false ∧ ∀ (j : Fin 64), j.val < i.val →
            validCell (text.toList[j.val]'(by omega)) = true := by
  rw [fromString_error_iff checked text side representable]
  change (text.toList.length = 64 ∧ ∃ sq,
    position = BitVec.ofNat 64 sq.val ∧ firstInvalid (Square.iter.zip text.toList) = some (sq,c)) ↔ _
  constructor
  · rintro ⟨len,sq,hp,hfind⟩
    obtain ⟨i,si,hc,bad,early⟩ := (firstInvalid_some_iff text.toList len sq c).mp hfind
    exact ⟨len,i,by simpa only [si] using hp,hc,bad,early⟩
  · rintro ⟨len,i,hp,hc,bad,early⟩
    refine ⟨len,i,hp,?_⟩
    exact (firstInvalid_some_iff text.toList len _ c).mpr ⟨i,rfl,hc,bad,early⟩

def FromStringContract : Prop :=
  ∀ (checked : Bool) (text : String) (side : Disc.PlayerColor), RepresentableText text →
    ((∃ result, fromString checked text side = .ok result) ↔
      text.toList.length = 64 ∧ ∀ c ∈ text.toList, c = 'X' ∨ c = 'O' ∨ c = '-') ∧
    (∀ result, fromString checked text side = .ok result → wellFormed result ∧
      ∃ len : text.toList.length = 64, ∀ i : Fin 64,
        result.player.getLsbD i.val = decide (text.toList[i.val]'(by omega) = side.toChar) ∧
        result.opponent.getLsbD i.val = decide (text.toList[i.val]'(by omega) = side.opposite.toChar)) ∧
    (∀ reason, fromString checked text side = .error (.error reason) ↔ textErrorSpec text reason) ∧
    (∀ c position, fromString checked text side = .error (.error (.invalidChar c position)) ↔
      ∃ len : text.toList.length = 64,
        ∃ (i : Fin 64), position = BitVec.ofNat 64 i.val ∧ text.toList[i.val]'(by omega) = c ∧
          validCell c = false ∧ ∀ (j : Fin 64), j.val < i.val →
            validCell (text.toList[j.val]'(by omega)) = true) ∧
    (∀ message, fromString checked text side ≠ .error (.panic message))

theorem fromStringContract_proven : FromStringContract := by
  intro checked text side representable
  refine ⟨?_,?_,fromString_error_iff checked text side representable,
    fromString_invalidChar_iff checked text side representable,
    fromString_no_panic checked text side representable⟩
  · simpa only [validCell_iff] using fromString_success_iff checked text side representable
  · intro result success
    exact ⟨fromString_disjoint checked text side representable result success,
      fromString_success_bits checked text side representable result success⟩

end NeuralReversi.Board
