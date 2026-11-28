module
public import NeuralReversi.Square.TextProperties
public import NeuralReversi.Square.SequenceModel

@[expose] public section

namespace NeuralReversi.Square
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000

def rawBytes (text : String) : List Byte :=
  text.toUTF8.data.toList.map fun b => BitVec.ofNat 8 b.toNat

theorem rawBytes_length (text : String) : (rawBytes text).length = text.toByteArray.size := by
  simp only [rawBytes, List.length_map, Array.length_toList, String.toUTF8_eq_toByteArray]
  rfl

theorem trimmedBytes_full_size (text : String)
    (bound : text.toByteArray.size ≤ 2) (full : (trimmedBytes text).length = 2) :
    trimWhitespace text.toList = text.toList := by
  obtain ⟨pre,post,split,_,_⟩ := trimWhitespace_decomposition text.toList
  have total := congrArg (fun chars : List Char => chars.utf8Encode.size) split
  simp only [String.utf8Encode_toList, List.utf8Encode_append, ByteArray.size_append] at total
  have full' : (trimWhitespace text.toList).utf8Encode.size = 2 := by
    simpa only [trimmedBytes, List.length_map, Array.length_toList, String.toUTF8_eq_toByteArray, String.toByteArray_ofList, ← ByteArray.size_data] using full
  have pzero : pre.utf8Encode.size = 0 := by omega
  have qzero : post.utf8Encode.size = 0 := by omega
  have pe : pre = [] := List.utf8Encode_eq_empty.mp (ByteArray.size_eq_zero_iff.mp pzero)
  have qe : post = [] := List.utf8Encode_eq_empty.mp (ByteArray.size_eq_zero_iff.mp qzero)
  simpa only [pe,qe,List.nil_append,List.append_nil] using split.symm

theorem fromStr_small_success (text : String) (sq : Square)
    (bound : text.toByteArray.size ≤ 2) (success : fromStr text = .ok sq) :
    ∃ f r, rawBytes text = [f,r] ∧ validFile f ∧ validRank r ∧
      sq.val = 8 * (r.toNat - 49) + fileNumber f := by
  obtain ⟨f,r,he,hf,hr,hi⟩ := (fromStr_success text sq).mp success
  have full : (trimmedBytes text).length = 2 := by rw [he]; rfl
  have trimEq := trimmedBytes_full_size text bound full
  have rawEq : trimmedBytes text = rawBytes text := by
    simp only [trimmedBytes, trimEq, String.ofList_toList, rawBytes]
  exact ⟨f,r,rawEq ▸ he,hf,hr,hi⟩

theorem byteArray_size (bytes : List Byte) : (byteArray bytes).size = bytes.length := by
  simp only [byteArray, ByteArray.size, List.size_toArray, List.length_map]

theorem fromUTF8_some_bytes (bytes : ByteArray) (text : String)
    (decoded : String.fromUTF8? bytes = some text) : text.toByteArray = bytes := by
  unfold String.fromUTF8? at decoded
  split at decoded
  · cases decoded
    rfl
  · simp at decoded

theorem rawBytes_of_decode (bytes : List Byte) (text : String)
    (decoded : String.fromUTF8? (byteArray bytes) = some text) : rawBytes text = bytes := by
  have exactBytes := fromUTF8_some_bytes _ _ decoded
  simp only [rawBytes, String.toUTF8_eq_toByteArray, exactBytes, byteArray, List.toList_toArray, List.map_map]
  simp [Function.comp_def]

theorem parseChunk_success_forward (bytes : List Byte) (bound : bytes.length ≤ 2) (sq : Square)
    (success : parseChunk bytes = .ok sq) :
    ∃ f r, bytes = [f,r] ∧ validFile f ∧ validRank r ∧ sq.val = 8 * (r.toNat - 49) + fileNumber f := by
  unfold parseChunk decodeByteList at success
  cases decoded : String.fromUTF8? (byteArray bytes) with
  | none => simp [decoded] at success
  | some text =>
    simp only [decoded] at success
    have textBound : text.toByteArray.size ≤ 2 := by
      rw [fromUTF8_some_bytes _ _ decoded, byteArray_size]
      exact bound
    simpa only [rawBytes_of_decode bytes text decoded] using fromStr_small_success text sq textBound success

theorem parseChunk_no_panic (bytes : List Byte) (panic : Panic) :
    parseChunk bytes ≠ .error (.panic panic) := by
  unfold parseChunk decodeByteList
  cases String.fromUTF8? (byteArray bytes) with
  | none => simp
  | some text => exact fromStr_no_panic text panic

def byteChar (b : Byte) : Char := Char.ofUInt8 (UInt8.ofBitVec b)

theorem byteChar_nat (b : Byte) : (byteChar b).toNat = b.toNat := by
  simp [byteChar,Char.ofUInt8,Char.toNat]
  have bound := b.isLt
  omega

theorem byteChar_utf8 (b : Byte) (ascii : b.toNat < 128) :
    String.utf8EncodeChar (byteChar b) = [UInt8.ofBitVec b] := by
  have hsize : (byteChar b).utf8Size = 1 := by
    rw [Char.utf8Size_eq_one_iff, UInt32.le_iff_toNat_le]
    change (byteChar b).toNat ≤ 127
    rw [byteChar_nat]
    omega
  rw [String.utf8EncodeChar_eq_singleton hsize]
  simp [byteChar, Char.ofUInt8]

theorem valid_byteChar_not_whitespace (b : Byte) (good : validFile b ∨ validRank b) :
    rustWhitespace (byteChar b) = false := by
  simp only [rustWhitespace, byteChar_nat, decide_eq_false_iff_not]
  unfold validFile validRank at good
  omega

theorem valid_pair_encoded (f r : Byte) (hf : validFile f) (hr : validRank r) :
    [byteChar f,byteChar r].utf8Encode = byteArray [f,r] := by
  have af : f.toNat < 128 := by unfold validFile at hf; omega
  have ar : r.toNat < 128 := by unfold validRank at hr; omega
  rw [List.utf8Encode_cons]
  simp only [List.utf8Encode_singleton,
    byteChar_utf8 f af, byteChar_utf8 r ar]
  rfl

theorem decode_valid_pair (f r : Byte) (hf : validFile f) (hr : validRank r) :
    String.fromUTF8? (byteArray [f,r]) = some (String.ofList [byteChar f,byteChar r]) := by
  have valid : (byteArray [f,r]).IsValidUTF8 := ⟨[byteChar f,byteChar r], (valid_pair_encoded f r hf hr).symm⟩
  rw [String.fromUTF8?, dite_eq_left valid]
  congr 1
  apply String.toByteArray_inj.mp
  exact (valid_pair_encoded f r hf hr).symm

theorem parseChunk_success_reverse (f r : Byte) (sq : Square)
    (hf : validFile f) (hr : validRank r)
    (coord : sq.val = 8 * (r.toNat - 49) + fileNumber f) :
    parseChunk [f,r] = .ok sq := by
  rw [parseChunk, decodeByteList, decode_valid_pair f r hf hr]
  apply (fromStr_success _ sq).mpr
  refine ⟨f,r,?_,hf,hr,coord⟩
  have clean := trimWhitespace_padded_pair [] [] (byteChar f) (byteChar r)
    (by simp) (by simp) (valid_byteChar_not_whitespace f (.inl hf))
    (valid_byteChar_not_whitespace r (.inr hr))
  simp only [List.nil_append,List.append_nil] at clean
  have encoded := valid_pair_encoded f r hf hr
  simp only [trimmedBytes,String.toList_ofList,clean,String.toUTF8_eq_toByteArray,
    String.toByteArray_ofList,encoded,byteArray,List.map_cons,List.map_nil,List.toList_toArray]
  simp

theorem trimmedBytes_length_le (text : String) : (trimmedBytes text).length ≤ text.toUTF8.size := by
  obtain ⟨pre,post,split,_,_⟩ := trimWhitespace_decomposition text.toList
  have total := congrArg (fun chars : List Char => chars.utf8Encode.size) split
  simp only [String.utf8Encode_toList, List.utf8Encode_append, ByteArray.size_append] at total
  have len : (trimmedBytes text).length = (trimWhitespace text.toList).utf8Encode.size := by
    simp only [trimmedBytes,List.length_map,Array.length_toList,String.toUTF8_eq_toByteArray,
      String.toByteArray_ofList, ← ByteArray.size_data]
  rw [len,String.toUTF8_eq_toByteArray]
  omega

theorem sequenceDomain_trimmed (text : String) (domain : SequenceDomain text) :
    (trimmedBytes text).length < 2 ^ 64 := by
  have bound := trimmedBytes_length_le text
  exact Nat.lt_of_le_of_lt bound domain

theorem parseChunk_success_iff (bytes : List Byte) (bound : bytes.length ≤ 2) (sq : Square) :
    parseChunk bytes = .ok sq ↔
      ∃ f r, bytes = [f,r] ∧ validFile f ∧ validRank r ∧
        sq.val = 8 * (r.toNat - 49) + fileNumber f := by
  constructor
  · exact parseChunk_success_forward bytes bound sq
  · rintro ⟨f,r,rfl,hf,hr,coord⟩
    exact parseChunk_success_reverse f r sq hf hr coord

def ChunkErrorSpec (bytes : List Byte) (reason : ParseError) : Prop :=
  (decodeByteList bytes = none ∧ reason = .invalidFormat) ∨
  ∃ text, decodeByteList bytes = some text ∧ ByteErrorSpec (trimmedBytes text) reason

theorem parseChunk_error_iff (bytes : List Byte) (reason : ParseError) :
    parseChunk bytes = .error (.error reason) ↔ ChunkErrorSpec bytes reason := by
  unfold parseChunk ChunkErrorSpec
  cases hd : decodeByteList bytes with
  | none => cases reason <;> simp
  | some text => simpa [fromStr] using parseBytes_error (trimmedBytes text) reason

end NeuralReversi.Square
