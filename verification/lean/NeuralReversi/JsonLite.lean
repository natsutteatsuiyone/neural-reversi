module

public section

namespace NeuralReversi.JsonLite

inductive Value where
  | null
  | bool (b : Bool)
  | int (n : Int)
  | str (s : String)
  | arr (items : List Value)
  | obj (fields : List (String × Value))

private def hexDigit (n : Nat) : Char :=
  Char.ofNat (if n < 10 then 48 + n else 87 + n)

def escapeChar (c : Char) : List Char :=
  match c with
  | '"' => ['\\', '"']
  | '\\' => ['\\', '\\']
  | '\n' => ['\\', 'n']
  | '\r' => ['\\', 'r']
  | '\t' => ['\\', 't']
  | _ =>
    let n := c.toNat
    if n < 0x20 then
      ['\\', 'u', '0', '0', hexDigit (n / 16), hexDigit (n % 16)]
    else
      [c]

def quoteString (s : String) : String :=
  String.ofList ('"' :: s.toList.flatMap escapeChar ++ ['"'])

def renderStringList (xs : List String) : String :=
  "[" ++ String.intercalate "," (xs.map quoteString) ++ "]"

private def isWs (c : Char) : Bool :=
  c == ' ' || c == '\t' || c == '\n' || c == '\r'

def skipWs (cs : Array Char) (pos : Nat) : Nat → Nat
  | 0 => pos
  | fuel + 1 =>
    match cs[pos]? with
    | some c => if isWs c then skipWs cs (pos + 1) fuel else pos
    | none => pos

private def matchKeyword (cs : Array Char) (pos : Nat) : List Char → Option Nat
  | [] => some pos
  | expected :: rest =>
    match cs[pos]? with
    | some c => if c == expected then matchKeyword cs (pos + 1) rest else none
    | none => none

private def hexVal (c : Char) : Option Nat :=
  let n := c.toNat
  if 48 ≤ n && n ≤ 57 then some (n - 48)
  else if 97 ≤ n && n ≤ 102 then some (n - 87)
  else if 65 ≤ n && n ≤ 70 then some (n - 55)
  else none

private def parseUnicode4 (cs : Array Char) (pos : Nat) : Except String (Char × Nat) := do
  let some c0 := cs[pos]? | throw "unterminated \\u escape"
  let some c1 := cs[pos + 1]? | throw "unterminated \\u escape"
  let some c2 := cs[pos + 2]? | throw "unterminated \\u escape"
  let some c3 := cs[pos + 3]? | throw "unterminated \\u escape"
  let some h0 := hexVal c0 | throw "invalid hex digit in \\u escape"
  let some h1 := hexVal c1 | throw "invalid hex digit in \\u escape"
  let some h2 := hexVal c2 | throw "invalid hex digit in \\u escape"
  let some h3 := hexVal c3 | throw "invalid hex digit in \\u escape"
  let code := h0 * 4096 + h1 * 256 + h2 * 16 + h3
  return (Char.ofNat code, pos + 4)

private def parseStringChars (cs : Array Char) (pos : Nat) (acc : List Char) :
    Nat → Except String (String × Nat)
  | 0 => .error "unterminated JSON string"
  | fuel + 1 =>
    match cs[pos]? with
    | none => .error "unterminated JSON string"
    | some '"' => .ok (String.ofList acc.reverse, pos + 1)
    | some '\\' =>
      match cs[pos + 1]? with
      | none => .error "unterminated JSON escape"
      | some '"' => parseStringChars cs (pos + 2) ('"' :: acc) fuel
      | some '\\' => parseStringChars cs (pos + 2) ('\\' :: acc) fuel
      | some '/' => parseStringChars cs (pos + 2) ('/' :: acc) fuel
      | some 'b' => parseStringChars cs (pos + 2) (Char.ofNat 8 :: acc) fuel
      | some 'f' => parseStringChars cs (pos + 2) (Char.ofNat 12 :: acc) fuel
      | some 'n' => parseStringChars cs (pos + 2) ('\n' :: acc) fuel
      | some 'r' => parseStringChars cs (pos + 2) ('\r' :: acc) fuel
      | some 't' => parseStringChars cs (pos + 2) ('\t' :: acc) fuel
      | some 'u' => do
        let (ch, nextPos) ← parseUnicode4 cs (pos + 2)
        parseStringChars cs nextPos (ch :: acc) fuel
      | some _ => .error "invalid JSON escape sequence"
    | some c =>
      if c.toNat < 0x20 then .error "unescaped control character in JSON string"
      else parseStringChars cs (pos + 1) (c :: acc) fuel

private def parseString (cs : Array Char) (pos : Nat) : Except String (String × Nat) :=
  match cs[pos]? with
  | some '"' => parseStringChars cs (pos + 1) [] (cs.size - pos)
  | _ => .error s!"expected JSON string at position {pos}"

private def digitVal (c : Char) : Option Nat :=
  let n := c.toNat
  if 48 ≤ n && n ≤ 57 then some (n - 48) else none

private def parseDigits (cs : Array Char) (pos : Nat) (acc : Nat) : Nat → Nat × Nat
  | 0 => (acc, pos)
  | fuel + 1 =>
    match cs[pos]? with
    | some c =>
      match digitVal c with
      | some d => parseDigits cs (pos + 1) (acc * 10 + d) fuel
      | none => (acc, pos)
    | none => (acc, pos)

private def parseInt (cs : Array Char) (pos : Nat) : Except String (Int × Nat) := do
  let (neg, startPos) :=
    match cs[pos]? with
    | some '-' => (true, pos + 1)
    | _ => (false, pos)
  let some first := cs[startPos]? | throw s!"expected number at position {startPos}"
  let some d0 := digitVal first | throw s!"expected digit at position {startPos}"
  let (mag, endPos) ←
    if d0 == 0 then
      match cs[startPos + 1]? with
      | some next =>
        if (digitVal next).isSome then throw s!"leading zero in JSON number at position {startPos}"
        else pure (0, startPos + 1)
      | none => pure (0, startPos + 1)
    else
      pure (parseDigits cs (startPos + 1) d0 (cs.size - startPos))
  match cs[endPos]? with
  | some '.' | some 'e' | some 'E' =>
    throw s!"unsupported non-integer JSON number at position {endPos}"
  | _ =>
    let val : Int := if neg then -(Int.ofNat mag) else Int.ofNat mag
    pure (val, endPos)

mutual
  private def parseValue (cs : Array Char) (pos : Nat) : Nat → Except String (Value × Nat)
    | 0 => .error "JSON parse limit exceeded"
    | fuel + 1 =>
      let p := skipWs cs pos (cs.size - pos + 1)
      match cs[p]? with
      | none => .error "unexpected end of JSON input"
      | some 'n' =>
        match matchKeyword cs p ['n', 'u', 'l', 'l'] with
        | some next => .ok (.null, next)
        | none => .error s!"invalid JSON literal at position {p}"
      | some 't' =>
        match matchKeyword cs p ['t', 'r', 'u', 'e'] with
        | some next => .ok (.bool true, next)
        | none => .error s!"invalid JSON literal at position {p}"
      | some 'f' =>
        match matchKeyword cs p ['f', 'a', 'l', 's', 'e'] with
        | some next => .ok (.bool false, next)
        | none => .error s!"invalid JSON literal at position {p}"
      | some '"' => do
        let (s, next) ← parseString cs p
        pure (.str s, next)
      | some '[' =>
        let p1 := skipWs cs (p + 1) (cs.size - p)
        if cs[p1]? == some ']' then .ok (.arr [], p1 + 1)
        else parseArrayItems cs p1 [] fuel
      | some '{' =>
        let p1 := skipWs cs (p + 1) (cs.size - p)
        if cs[p1]? == some '}' then .ok (.obj [], p1 + 1)
        else parseObjFields cs p1 [] fuel
      | some c =>
        if c == '-' || (digitVal c).isSome then do
          let (n, next) ← parseInt cs p
          pure (.int n, next)
        else
          .error s!"unexpected character '{c}' at position {p}"

  private def parseArrayItems (cs : Array Char) (pos : Nat) (acc : List Value) :
      Nat → Except String (Value × Nat)
    | 0 => .error "JSON parse limit exceeded"
    | fuel + 1 => do
      let (item, afterItem) ← parseValue cs pos fuel
      let p := skipWs cs afterItem (cs.size - afterItem + 1)
      match cs[p]? with
      | some ',' => parseArrayItems cs (p + 1) (item :: acc) fuel
      | some ']' => .ok (.arr (item :: acc).reverse, p + 1)
      | _ => .error s!"expected ',' or ']' in JSON array at position {p}"

  private def parseObjFields (cs : Array Char) (pos : Nat) (acc : List (String × Value)) :
      Nat → Except String (Value × Nat)
    | 0 => .error "JSON parse limit exceeded"
    | fuel + 1 => do
      let pKey := skipWs cs pos (cs.size - pos + 1)
      let (key, afterKey) ← parseString cs pKey
      let pColon := skipWs cs afterKey (cs.size - afterKey + 1)
      unless cs[pColon]? == some ':' do
        throw s!"expected ':' after JSON key at position {pColon}"
      let (val, afterVal) ← parseValue cs (pColon + 1) fuel
      let pNext := skipWs cs afterVal (cs.size - afterVal + 1)
      match cs[pNext]? with
      | some ',' => parseObjFields cs (pNext + 1) ((key, val) :: acc) fuel
      | some '}' => .ok (.obj ((key, val) :: acc).reverse, pNext + 1)
      | _ => .error s!"expected ',' or '}}' in JSON object at position {pNext}"
end

def parse (input : String) : Except String Value := do
  let cs := input.toList.toArray
  let (val, pos) ← parseValue cs 0 (2 * cs.size + 4)
  let rest := skipWs cs pos (cs.size - pos + 1)
  if rest == cs.size then pure val
  else throw s!"trailing characters after JSON value at position {rest}"

class FromJson (α : Type) where
  fromJson? : Value → Except String α

export FromJson (fromJson?)

instance : FromJson Int where
  fromJson?
    | .int n => .ok n
    | _ => .error "expected JSON integer"

instance : FromJson Nat where
  fromJson?
    | .int n =>
      if 0 ≤ n then .ok n.toNat
      else .error s!"expected non-negative JSON integer, got {n}"
    | _ => .error "expected JSON natural number"

instance : FromJson Bool where
  fromJson?
    | .bool b => .ok b
    | _ => .error "expected JSON boolean"

instance : FromJson String where
  fromJson?
    | .str s => .ok s
    | _ => .error "expected JSON string"

instance [FromJson α] : FromJson (Option α) where
  fromJson?
    | .null => .ok none
    | v => some <$> fromJson? v

instance [FromJson α] : FromJson (List α) where
  fromJson?
    | .arr xs => xs.mapM fromJson?
    | _ => .error "expected JSON array"

def expectObj : Value → Except String (List (String × Value))
  | .obj fields => .ok fields
  | _ => .error "expected JSON object"

def lookupField (fields : List (String × Value)) (name : String) : Option Value :=
  (fields.reverse.find? (·.1 == name)).map Prod.snd

def getField [FromJson α] (fields : List (String × Value)) (name : String) : Except String α :=
  match lookupField fields name with
  | some v => fromJson? v
  | none => .error s!"property not found: {name}"

def getFieldOpt [FromJson α] (fields : List (String × Value)) (name : String) :
    Except String (Option α) :=
  match lookupField fields name with
  | none | some .null => .ok none
  | some v => some <$> fromJson? v

end NeuralReversi.JsonLite
