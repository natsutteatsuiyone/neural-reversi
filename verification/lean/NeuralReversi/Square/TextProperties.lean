module
public import NeuralReversi.Square.TextModel

@[expose] public section

namespace NeuralReversi.Square

def isAsciiPair (f r : Byte) : Prop := f.toNat < 128 ∧ r.toNat < 128

def validFile (b : Byte) : Prop :=
  (65 ≤ b.toNat ∧ b.toNat ≤ 72) ∨ (97 ≤ b.toNat ∧ b.toNat ≤ 104)

def fileNumber (b : Byte) : Nat :=
  if b.toNat ≤ 72 then b.toNat - 65 else b.toNat - 97

def validRank (b : Byte) : Prop := 49 ≤ b.toNat ∧ b.toNat ≤ 56

theorem valid_isAsciiPair (f r : Byte) (hf : validFile f) (hr : validRank r) :
    isAsciiPair f r := by
  unfold validFile at hf
  unfold validRank at hr
  unfold isAsciiPair
  omega

theorem lower_value (b : Byte) : (asciiLowerByte b).toNat =
    if 65 ≤ b.toNat ∧ b.toNat ≤ 90 then b.toNat + 32 else b.toNat := by
  unfold asciiLowerByte
  split <;> rename_i h
  · change (b.toNat + 32) % 256 = b.toNat + 32
    omega
  · rfl

theorem lower_valid (b : Byte) :
    (97 ≤ (asciiLowerByte b).toNat ∧ (asciiLowerByte b).toNat ≤ 104) ↔ validFile b := by
  rw [lower_value]
  unfold validFile
  split <;> omega

theorem parser_coordinates (f r : Byte) (hf : validFile f) (hr : validRank r) :
    (asciiLowerByte f - 97).toNat = fileNumber f ∧
    (r - 49).toNat = r.toNat - 49 ∧
    (asciiLowerByte f - 97).toNat < 8 ∧ (r - 49).toNat < 8 := by
  have hl := (lower_valid f).mpr hf
  have hfv : (asciiLowerByte f - 97).toNat = (asciiLowerByte f).toNat - 97 := by
    change (256 - 97 + (asciiLowerByte f).toNat) % 256 = _
    omega
  have hrv : (r - 49).toNat = r.toNat - 49 := by
    change (256 - 49 + r.toNat) % 256 = _
    unfold validRank at hr
    omega
  rw [hfv, hrv]
  refine ⟨?_, rfl, by omega, ?_⟩
  · rw [lower_value]
    unfold validFile at hf
    unfold fileNumber
    split <;> split <;> omega
  · unfold validRank at hr
    omega

theorem parse_pair_valid (f r : Byte) (hf : validFile f) (hr : validRank r) :
    ∃ sq, parseBytes [f,r] = .ok sq ∧
      sq.val = 8 * (r.toNat - 49) + fileNumber f ∧ sq.val < 64 := by
  obtain ⟨hfv, hrv, hfb, hrb⟩ := parser_coordinates f r hf hr
  have hl := (lower_valid f).mpr hf
  have ha : f.toNat < 128 ∧ r.toNat < 128 := valid_isAsciiPair f r hf hr
  let sq : Square := ⟨(r - 49).toNat * 8 + (asciiLowerByte f - 97).toNat, by omega⟩
  refine ⟨sq, ?_, ?_, ?_⟩
  · have hc : fromFileRank (asciiLowerByte f - 97) (r - 49) = .ok sq := by
      unfold fromFileRank
      rw [dite_eq_left hfb, dite_eq_left hrb]
    change 49 ≤ r.toNat ∧ r.toNat ≤ 56 at hr
    simp only [parseBytes, ite_eq_right (not_not_intro ha), ite_eq_right (not_not_intro hl),
      ite_eq_right (not_not_intro hr), hc, Except.mapError]
  · change (r - 49).toNat * 8 + (asciiLowerByte f - 97).toNat = _
    rw [hfv, hrv]
    omega
  · change (r - 49).toNat * 8 + (asciiLowerByte f - 97).toNat < 64
    omega

theorem parse_pair_non_ascii (f r : Byte) (ha : ¬ isAsciiPair f r) :
    parseBytes [f,r] = .error (.error .invalidFormat) := by
  change ¬ (f.toNat < 128 ∧ r.toNat < 128) at ha
  simp only [parseBytes, ite_eq_left ha]

theorem parse_pair_bad_file (f r : Byte) (ha : isAsciiPair f r) (hf : ¬ validFile f) :
    parseBytes [f,r] = .error (.error (.invalidFile f)) := by
  have hl : ¬ (97 ≤ (asciiLowerByte f).toNat ∧ (asciiLowerByte f).toNat ≤ 104) :=
    fun h => hf ((lower_valid f).mp h)
  change f.toNat < 128 ∧ r.toNat < 128 at ha
  simp only [parseBytes, ite_eq_right (not_not_intro ha), ite_eq_left hl]

theorem parse_pair_bad_rank (f r : Byte) (ha : isAsciiPair f r) (hf : validFile f) (hr : ¬ validRank r) :
    parseBytes [f,r] = .error (.error (.invalidRank r)) := by
  have hl := (lower_valid f).mpr hf
  change f.toNat < 128 ∧ r.toNat < 128 at ha
  change ¬ (49 ≤ r.toNat ∧ r.toNat ≤ 56) at hr
  simp only [parseBytes, ite_eq_right (not_not_intro ha), ite_eq_right (not_not_intro hl), ite_eq_left hr]

theorem parse_pair_success (f r : Byte) (sq : Square) :
    parseBytes [f,r] = .ok sq ↔ validFile f ∧ validRank r ∧
      sq.val = 8 * (r.toNat - 49) + fileNumber f := by
  by_cases ha : isAsciiPair f r
  · by_cases hf : validFile f
    · by_cases hr : validRank r
      · obtain ⟨v, hv, hi, _⟩ := parse_pair_valid f r hf hr
        rw [hv]
        simp only [Except.ok.injEq, hf, hr, true_and]
        constructor
        · intro h; subst sq; exact hi
        · intro h; apply Fin.ext; omega
      · rw [parse_pair_bad_rank f r ha hf hr]
        simp [hr]
    · rw [parse_pair_bad_file f r ha hf]
      simp [hf]
  · rw [parse_pair_non_ascii f r ha]
    constructor
    · intro h; cases h
    · rintro ⟨hf, hr, _⟩
      exact False.elim (ha (valid_isAsciiPair f r hf hr))

theorem parse_pair_no_panic (f r : Byte) (panic : Panic) :
    parseBytes [f,r] ≠ .error (.panic panic) := by
  by_cases ha : isAsciiPair f r
  · by_cases hf : validFile f
    · by_cases hr : validRank r
      · obtain ⟨sq, hs, _, _⟩ := parse_pair_valid f r hf hr
        rw [hs]; simp
      · rw [parse_pair_bad_rank f r ha hf hr]; simp
    · rw [parse_pair_bad_file f r ha hf]; simp
  · rw [parse_pair_non_ascii f r ha]; simp

theorem parseBytes_no_panic (bytes : List Byte) (panic : Panic) :
    parseBytes bytes ≠ .error (.panic panic) := by
  cases bytes with
  | nil => simp [parseBytes]
  | cons f tail =>
    cases tail with
    | nil => simp [parseBytes]
    | cons r tail =>
      cases tail with
      | nil => exact parse_pair_no_panic f r panic
      | cons x tail => simp [parseBytes]

theorem parseBytes_success (bytes : List Byte) (sq : Square) :
    parseBytes bytes = .ok sq ↔
      ∃ f r, bytes = [f,r] ∧ validFile f ∧ validRank r ∧
        sq.val = 8 * (r.toNat - 49) + fileNumber f := by
  cases bytes with
  | nil => simp [parseBytes]
  | cons f tail =>
    cases tail with
    | nil => simp [parseBytes]
    | cons r tail =>
      cases tail with
      | nil =>
        constructor
        · intro h
          exact ⟨f, r, rfl, (parse_pair_success f r sq).mp h⟩
        · rintro ⟨f2, r2, he, hv⟩
          rcases List.cons.inj he with ⟨rfl, ht⟩
          rcases List.cons.inj ht with ⟨rfl, _⟩
          exact (parse_pair_success f r sq).mpr hv
      | cons x tail => simp [parseBytes]


theorem parseBytes_bad_length (bytes : List Byte) (hlen : bytes.length ≠ 2) :
    parseBytes bytes = .error (.error .invalidFormat) := by
  cases bytes with
  | nil => rfl
  | cons f tail =>
    cases tail with
    | nil => rfl
    | cons r tail =>
      cases tail with
      | nil => simp at hlen
      | cons x tail => rfl

theorem fromStr_no_panic (text : String) (panic : Panic) :
    fromStr text ≠ .error (.panic panic) :=
  parseBytes_no_panic (trimmedBytes text) panic

theorem fromStr_success (text : String) (sq : Square) :
    fromStr text = .ok sq ↔
      ∃ f r, trimmedBytes text = [f,r] ∧ validFile f ∧ validRank r ∧
        sq.val = 8 * (r.toNat - 49) + fileNumber f :=
  parseBytes_success (trimmedBytes text) sq

def ByteErrorSpec (bytes : List Byte) (reason : ParseError) : Prop :=
  match bytes with
  | [f, r] =>
      (¬ isAsciiPair f r ∧ reason = .invalidFormat) ∨
      (isAsciiPair f r ∧ ¬ validFile f ∧ reason = .invalidFile f) ∨
      (isAsciiPair f r ∧ validFile f ∧ ¬ validRank r ∧ reason = .invalidRank r)
  | _ => reason = .invalidFormat

theorem parse_pair_error (f r : Byte) (reason : ParseError) :
    parseBytes [f, r] = .error (.error reason) ↔
      (¬ isAsciiPair f r ∧ reason = .invalidFormat) ∨
      (isAsciiPair f r ∧ ¬ validFile f ∧ reason = .invalidFile f) ∨
      (isAsciiPair f r ∧ validFile f ∧ ¬ validRank r ∧ reason = .invalidRank r) := by
  by_cases ha : isAsciiPair f r
  · by_cases hf : validFile f
    · by_cases hr : validRank r
      · obtain ⟨sq, hs, _, _⟩ := parse_pair_valid f r hf hr
        rw [hs]
        simp [ha, hf, hr]
      · rw [parse_pair_bad_rank f r ha hf hr]
        simp [ha, hf, hr, eq_comm]
    · rw [parse_pair_bad_file f r ha hf]
      simp [ha, hf, eq_comm]
  · rw [parse_pair_non_ascii f r ha]
    simp [ha, eq_comm]

example : fromStr "é" = .error (.error .invalidFormat) := by rfl

theorem parseBytes_error (bytes : List Byte) (reason : ParseError) :
    parseBytes bytes = .error (.error reason) ↔ ByteErrorSpec bytes reason := by
  cases bytes with
  | nil => cases reason <;> simp [parseBytes, ByteErrorSpec]
  | cons f tail =>
    cases tail with
    | nil => cases reason <;> simp [parseBytes, ByteErrorSpec]
    | cons r tail =>
      cases tail with
      | nil => exact parse_pair_error f r reason
      | cons x tail => cases reason <;> simp [parseBytes, ByteErrorSpec]

theorem fromStr_playable (_text : String) (sq : Square) (_success : fromStr _text = .ok sq) :
    sq.val < 64 :=
  sq.isLt

def TextSyntaxContract : Prop :=
  (∀ (text : String) (sq : Square), fromStr text = .ok sq ↔
    ∃ f r, trimmedBytes text = [f, r] ∧ validFile f ∧ validRank r ∧
      sq.val = 8 * (r.toNat - 49) + fileNumber f) ∧
  ∀ (text : String) (sq : Square), fromStr text = .ok sq → sq.val < 64

theorem textSyntaxContract_proven : TextSyntaxContract :=
  ⟨fromStr_success, fromStr_playable⟩

def TextErrorContract : Prop :=
  (∀ (text : String) (reason : ParseError),
    fromStr text = .error (.error reason) ↔ ByteErrorSpec (trimmedBytes text) reason) ∧
  ∀ (text : String) (reason : Panic), fromStr text ≠ .error (.panic reason)

theorem textErrorContract_proven : TextErrorContract :=
  ⟨fun text reason => parseBytes_error (trimmedBytes text) reason, fromStr_no_panic⟩

private theorem rustWhitespace_range (n : Nat) :
    (0x09 ≤ n ∧ n ≤ 0x0d) ↔ n = 0x09 ∨ n = 0x0a ∨ n = 0x0b ∨ n = 0x0c ∨ n = 0x0d := by omega

private theorem rustWhitespace_range2 (n : Nat) :
    (0x2000 ≤ n ∧ n ≤ 0x200a) ↔
      n = 0x2000 ∨ n = 0x2001 ∨ n = 0x2002 ∨ n = 0x2003 ∨ n = 0x2004 ∨ n = 0x2005 ∨
      n = 0x2006 ∨ n = 0x2007 ∨ n = 0x2008 ∨ n = 0x2009 ∨ n = 0x200a := by omega

theorem rustWhitespace_table (c : Char) :
    rustWhitespace c = true ↔ c.toNat ∈ unicodeWhitespaceScalars := by
  simp only [rustWhitespace, decide_eq_true_eq,
    unicodeWhitespaceScalars, List.mem_cons, List.not_mem_nil, or_false,
    rustWhitespace_range, rustWhitespace_range2, or_assoc]

theorem trimWhitespace_decomposition (xs : List Char) :
    ∃ pre post, xs = pre ++ trimWhitespace xs ++ post ∧
      (∀ c ∈ pre, rustWhitespace c = true) ∧ (∀ c ∈ post, rustWhitespace c = true) := by
  let pre := xs.takeWhile rustWhitespace
  let rest := xs.dropWhile rustWhitespace
  let post := (rest.reverse.takeWhile rustWhitespace).reverse
  refine ⟨pre, post, ?_, ?_, ?_⟩
  · have h := congrArg List.reverse
      (List.takeWhile_append_dropWhile (p := rustWhitespace) (l := rest.reverse))
    simp only [List.reverse_append, List.reverse_reverse] at h
    have hx := List.takeWhile_append_dropWhile (p := rustWhitespace) (l := xs)
    change pre ++ rest = xs at hx
    change xs = (pre ++ (rest.reverse.dropWhile rustWhitespace).reverse) ++ post
    calc
      xs = pre ++ rest := hx.symm
      _ = (pre ++ (rest.reverse.dropWhile rustWhitespace).reverse) ++ post := by
        rw [List.append_assoc]
        exact congrArg (pre ++ ·) h.symm
  · intro a ha
    exact List.all_eq_true.mp (List.all_takeWhile (p := rustWhitespace) (l := xs)) a ha
  · intro a ha
    exact List.all_eq_true.mp
      (List.all_takeWhile (p := rustWhitespace) (l := rest.reverse)) a (List.mem_reverse.mp ha)

theorem trimWhitespace_padded_pair (pre post : List Char) (a b : Char)
    (hp : ∀ c ∈ pre, rustWhitespace c = true)
    (hs : ∀ c ∈ post, rustWhitespace c = true)
    (ha : rustWhitespace a = false) (hb : rustWhitespace b = false) :
    trimWhitespace (pre ++ [a, b] ++ post) = [a, b] := by
  unfold trimWhitespace
  rw [List.append_assoc, List.dropWhile_append_of_pos hp]
  simp only [List.cons_append, List.nil_append, List.dropWhile_cons, ha,
    Bool.false_eq_true, ↓reduceIte, List.reverse_cons, List.nil_append]
  have hsr : ∀ c ∈ post.reverse, rustWhitespace c = true := by
    intro c hc
    exact hs c (List.mem_reverse.mp hc)
  rw [List.append_assoc, List.dropWhile_append_of_pos hsr]
  simp [hb]

private theorem getLast_dropWhile_inherited (p : α → Bool) (xs : List α) (a : α)
    (h : (xs.dropWhile p).getLast? = some a) : xs.getLast? = some a := by
  calc
    xs.getLast? = (xs.takeWhile p ++ xs.dropWhile p).getLast? :=
      congrArg List.getLast? List.takeWhile_append_dropWhile.symm
    _ = some a := by rw [List.getLast?_append, h]; rfl

theorem trimWhitespace_head_not (xs : List Char) (c : Char)
    (h : (trimWhitespace xs).head? = some c) : rustWhitespace c = false := by
  simp only [trimWhitespace, List.head?_reverse] at h
  have inherited := getLast_dropWhile_inherited rustWhitespace
    (xs.dropWhile rustWhitespace).reverse c h
  simp only [List.getLast?_reverse] at inherited
  have clean := List.head?_dropWhile_not rustWhitespace xs
  simpa only [inherited] using clean

theorem trimWhitespace_last_not (xs : List Char) (c : Char)
    (h : (trimWhitespace xs).getLast? = some c) : rustWhitespace c = false := by
  simp only [trimWhitespace, List.getLast?_reverse] at h
  have clean := List.head?_dropWhile_not rustWhitespace (xs.dropWhile rustWhitespace).reverse
  simpa only [h] using clean

private theorem dropWhile_eq_self_of_clean_head (p : α → Bool) (xs : List α)
    (clean : ∀ a, xs.head? = some a → p a = false) : xs.dropWhile p = xs := by
  cases xs with
  | nil => rfl
  | cons a tail =>
    have ha := clean a rfl
    simp [ha]

theorem trimWhitespace_idempotent (xs : List Char) :
    trimWhitespace (trimWhitespace xs) = trimWhitespace xs := by
  have hl := dropWhile_eq_self_of_clean_head rustWhitespace (trimWhitespace xs)
    (trimWhitespace_head_not xs)
  have hr : (trimWhitespace xs).reverse.dropWhile rustWhitespace =
      (trimWhitespace xs).reverse := by
    apply dropWhile_eq_self_of_clean_head
    intro c hc
    simp only [List.head?_reverse] at hc
    exact trimWhitespace_last_not xs c hc
  change (((trimWhitespace xs).dropWhile rustWhitespace).reverse.dropWhile rustWhitespace).reverse = _
  rw [hl, hr, List.reverse_reverse]

theorem fromStr_trimmed (text : String) :
    fromStr (String.ofList (trimWhitespace text.toList)) = fromStr text := by
  simp only [fromStr, trimmedBytes, String.toList_ofList, trimWhitespace_idempotent]

end NeuralReversi.Square
