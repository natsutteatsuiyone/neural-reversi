module
public import NeuralReversi.Board.ParseModel
public import NeuralReversi.Square.TextModel

@[expose] public section

namespace NeuralReversi.Obf

open Board (Board)
open Disc (PlayerColor)

abbrev MoveScore := Square.Square × Int

structure Position where
  board : Board
  side : PlayerColor
  moves : List MoveScore
  passScore : Option Int

inductive ScoreFailure where
  | empty
  | invalidDigit
  | positiveOverflow
  | negativeOverflow
  deriving DecidableEq

inductive Error where
  | nonAsciiHeader (header : String)
  | shortHeader (header : String)
  | invalidSide (side : String)
  | invalidBoard (failure : Board.TextFailure)
  | invalidSegment (segment : String)
  | invalidMove (input : String) (failure : Square.ParseFailure)
  | invalidScore (input : String) (failure : ScoreFailure)
  deriving DecidableEq

def trim (input : String) : String := String.ofList (Square.trimWhitespace input.toList)

def beforeComment (input : String) : String :=
  String.ofList (input.toList.takeWhile (· != '%'))

def splitSegment (segment : String) : Option (String × String) :=
  let key := segment.toList.takeWhile (· != ':')
  match segment.toList.dropWhile (· != ':') with
  | [] => none
  | _ :: score => some (String.ofList key, String.ofList score)

def parseHeader (header : String) : Except Error (Board × PlayerColor) :=
  let chars := header.toList
  if !(chars.all fun c => decide (c.toNat < 128)) then .error (.nonAsciiHeader header)
  else if chars.length < 65 then .error (.shortHeader header)
  else
    let label := trim (String.ofList (chars.drop 64))
    let side : Except Error PlayerColor := match label with
      | "X" => .ok .black
      | "O" => .ok .white
      | "" => .error (.shortHeader header)
      | other => .error (.invalidSide other)
    side.bind fun color =>
      let parsed := Except.mapError Error.invalidBoard (Board.fromString true (String.ofList (chars.take 64)) color)
      parsed.map fun board => (board, color)

def appendDigit (value : Nat) (digit : Char) : Nat := 10 * value + (digit.toNat - 48)

def readDigits (limit : Nat) (overflow : ScoreFailure) (value : Nat) :
    List Char → Except ScoreFailure Nat
  | [] => .ok value
  | c :: rest =>
    if c.isDigit then
      let next := appendDigit value c
      if next ≤ limit then readDigits limit overflow next rest else .error overflow
    else .error .invalidDigit

def parseInteger : List Char → Except ScoreFailure Int
  | [] => .error .empty
  | first :: digits =>
    if first = '-' then
      if digits.isEmpty then .error .invalidDigit
      else (readDigits 2147483648 .negativeOverflow 0 digits).map (fun n => -Int.ofNat n)
    else (readDigits 2147483647 .positiveOverflow 0 (first :: digits)).map Int.ofNat

def parseScore (input : String) : Except Error Int :=
  (parseInteger ((trim input).toList.dropWhile (· == '+'))).mapError (Error.invalidScore input)

def ScoreFailure.message : ScoreFailure → String
  | .empty => "cannot parse integer from empty string"
  | .invalidDigit => "invalid digit found in string"
  | .positiveOverflow => "number too large to fit in target type"
  | .negativeOverflow => "number too small to fit in target type"

def boardErrorMessage : Board.TextFailure → String
  | .error (.tooShort expected actual) =>
    s!"Board string too short: expected {expected.toNat} characters, got {actual.toNat}"
  | .error (.tooLong expected actual) =>
    s!"Board string too long: expected {expected.toNat} characters, got {actual.toNat}"
  | .error (.invalidChar c position) =>
    s!"Invalid character '{c}' at position {position.toNat}: must be 'X', 'O', or '-'"
  | .panic message => message

def squareErrorMessage : Square.ParseFailure → String
  | .error .invalidFormat => "Invalid square format: must be 2 ASCII characters (e.g., 'a1')"
  | .error (.invalidFile b) => s!"Invalid file '{Char.ofNat b.toNat}': must be a-h or A-H"
  | .error (.invalidRank b) => s!"Invalid rank '{Char.ofNat b.toNat}': must be 1-8"
  | .panic reason => reason.message

def Error.sourceMessage : Error → Option String
  | .invalidBoard failure => some (boardErrorMessage failure)
  | .invalidMove _ failure => some (squareErrorMessage failure)
  | .invalidScore _ failure => some failure.message
  | _ => none

def Error.message : Error → String
  | .nonAsciiHeader header => s!"Board header contains non-ASCII characters: '{header}'"
  | .shortHeader header => s!"Board header too short (need 64 board chars + side): '{header}'"
  | .invalidSide side => s!"Invalid side to move: '{side}'"
  | .invalidSegment segment => s!"Invalid move:score format: '{segment}'"
  | .invalidBoard failure => "Invalid board string: " ++ boardErrorMessage failure
  | .invalidMove input failure => s!"Invalid move '{input}': " ++ squareErrorMessage failure
  | .invalidScore input failure => s!"Invalid score '{input}': " ++ failure.message

def Error.messagePrefix : Error → Option String
  | .invalidBoard _ => some "Invalid board string: "
  | .invalidMove input _ => some s!"Invalid move '{input}': "
  | .invalidScore input _ => some s!"Invalid score '{input}': "
  | _ => none

def ErrorContract : Prop :=
  ∀ (error : Error),
    (error.sourceMessage.isSome = error.messagePrefix.isSome) ∧
    ∀ source, error.sourceMessage = some source →
      ∃ lead, error.messagePrefix = some lead ∧ error.message = lead ++ source

theorem errorContract_proven : ErrorContract := by
  intro error
  cases error <;> simp [Error.sourceMessage, Error.messagePrefix, Error.message]

inductive Segment where
  | ignored
  | pass (score : Int)
  | move (value : MoveScore)

def parseSegment (input : String) : Except Error Segment :=
  let segment := trim input
  if segment.isEmpty then .ok .ignored
  else match splitSegment segment with
  | none => .error (.invalidSegment segment)
  | some (label, value) =>
    let key := trim label
    if key == "--" then .ok .ignored
    else if ["PS", "Ps", "pS", "ps"].contains key then (parseScore value).map Segment.pass
    else
      ((Square.fromStr key).mapError (Error.invalidMove key)).bind fun square =>
        (parseScore value).map (fun score => .move (square, score))

def moveOrder (a b : MoveScore) : Bool := decide (b.2 ≤ a.2)

def orderMoves (scores : List MoveScore) : List MoveScore := scores.mergeSort moveOrder

def segmentMoves (entries : List Segment) : List MoveScore :=
  entries.filterMap fun
    | .move data => some data
    | _ => none

def lastPass (entries : List Segment) : Option Int :=
  entries.reverse.findSome? fun
    | .pass score => some score
    | _ => none

def parse (input : String) : Except Error (Option Position) :=
  let stripped := trim (beforeComment input)
  if stripped.isEmpty then .ok none
  else
    match stripped.splitOn ";" with
    | [] => .ok none
    | header :: segments =>
      (parseHeader (trim header)).bind fun (board, side) =>
        (segments.mapM parseSegment).map fun entries =>
          some ⟨board, side, orderMoves (segmentMoves entries), lastPass entries⟩

def isPass (pos : Position) : Bool := pos.passScore.isSome && pos.moves.isEmpty

def expectedScore (pos : Position) : Option Int :=
  (pos.moves.head?.map Prod.snd).or pos.passScore

def scoreOf (pos : Position) (square : Square.Square) : Option Int :=
  (pos.moves.find? fun entry => entry.1 == square).map Prod.snd

def rankFrom (square : Square.Square) (rank : Nat) : Nat → List MoveScore → Option Nat
  | 0, _ | _, [] => none
  | remaining + 1, first :: rest =>
    let equalScore := fun entry : MoveScore => entry.2 == first.2
    if (first :: rest.takeWhile equalScore).any (fun entry => entry.1 == square) then some rank
    else rankFrom square (rank + 1) remaining (rest.dropWhile equalScore)

def rankOf (pos : Position) (square : Square.Square) : Option Nat :=
  rankFrom square 0 3 pos.moves

def bestMoves (pos : Position) : List Square.Square :=
  let best := pos.moves.head?.map Prod.snd
  (pos.moves.takeWhile (fun entry => some entry.2 == best)).map Prod.fst

private theorem order_trans (a b c : MoveScore) (ab : moveOrder a b) (bc : moveOrder b c) :
    moveOrder a c := by
  simp only [moveOrder, decide_eq_true_eq] at ab bc ⊢
  omega

private theorem order_total (a b : MoveScore) : moveOrder a b || moveOrder b a := by
  simp only [Bool.or_eq_true, moveOrder, decide_eq_true_eq]
  omega

def SortContract : Prop :=
  ∀ scores : List MoveScore,
    (orderMoves scores).Perm scores ∧ (orderMoves scores).Pairwise (fun a b => moveOrder a b = true) ∧
      ∀ a b : MoveScore, a.2 = b.2 → [a, b].Sublist scores → [a, b].Sublist (orderMoves scores)

theorem sortContract_proven : SortContract := by
  intro scores
  refine ⟨List.mergeSort_perm _ _, List.pairwise_mergeSort order_trans order_total scores, ?_⟩
  intro a b equal original
  exact List.pair_sublist_mergeSort order_trans order_total (by simp [moveOrder, equal]) original

def ScoreContract : Prop :=
  ∀ input score, parseScore input = .ok score → -2147483648 ≤ score ∧ score ≤ 2147483647

private theorem readDigits_spec (limit : Nat) (overflow : ScoreFailure) (input : List Char)
    (start answer : Nat) (bounded : start ≤ limit)
    (run : readDigits limit overflow start input = .ok answer) :
    answer ≤ limit ∧ input.all Char.isDigit = true ∧ answer = input.foldl appendDigit start := by
  induction input generalizing start with
  | nil =>
    have same : start = answer := by simpa [readDigits] using run
    simpa [← same] using bounded
  | cons head tail ih =>
    simp only [readDigits] at run
    split at run
    next digit =>
      split at run
      next fits =>
        have ⟨upper, valid, equal⟩ := ih (appendDigit start head) fits run
        exact ⟨upper, by simp [digit, valid], equal⟩
      next => cases run
    next => cases run

private theorem readDigits_append (limit : Nat) (overflow : ScoreFailure) (start : Nat)
    (accepted pending : List Char) :
    readDigits limit overflow start (accepted ++ pending) =
      (readDigits limit overflow start accepted).bind (fun value => readDigits limit overflow value pending) := by
  induction accepted generalizing start with
  | nil => rfl
  | cons c rest ih =>
    simp only [List.cons_append, readDigits]
    split
    · split
      · exact ih _
      · rfl
    · rfl

private theorem readDigits_error (limit : Nat) (overflow error : ScoreFailure) (start : Nat)
    (input : List Char) (run : readDigits limit overflow start input = .error error) :
    ∃ front c back value, input = front ++ c :: back ∧ readDigits limit overflow start front = .ok value ∧
      (if c.isDigit then limit < appendDigit value c ∧ error = overflow else error = .invalidDigit) := by
  induction input generalizing start with
  | nil => cases run
  | cons c rest ih =>
    by_cases valid : c.isDigit = true
    · by_cases fits : appendDigit start c ≤ limit
      · have tail : readDigits limit overflow (appendDigit start c) rest = .error error := by
          simpa [readDigits, valid, fits] using run
        obtain ⟨front, bad, back, value, rfl, accepted, stopped⟩ := ih _ tail
        exact ⟨c :: front, bad, back, value, rfl, by simpa [readDigits, valid, fits] using accepted, stopped⟩
      · have same : overflow = error := by simpa [readDigits, valid, fits] using run
        exact ⟨[], c, rest, start, rfl, rfl, by simp [valid, ← same, Nat.lt_of_not_ge fits]⟩
    · have same : ScoreFailure.invalidDigit = error := by simpa [readDigits, valid] using run
      exact ⟨[], c, rest, start, rfl, rfl, by simp [valid, ← same]⟩

def ScoreDigitContract : Prop :=
  ∀ (negative : Bool) input,
    let limit := if negative then 2147483648 else 2147483647
    let overflow := if negative then ScoreFailure.negativeOverflow else ScoreFailure.positiveOverflow
    (∀ result, readDigits limit overflow 0 input = .ok result →
      result ≤ limit ∧ input.all Char.isDigit = true ∧ result = input.foldl appendDigit 0) ∧
    ∀ error, readDigits limit overflow 0 input = .error error ↔
      ∃ front c back value, input = front ++ c :: back ∧ readDigits limit overflow 0 front = .ok value ∧
        (if c.isDigit then limit < appendDigit value c ∧ error = overflow else error = .invalidDigit)

theorem scoreDigitContract_proven : ScoreDigitContract := by
  intro negative input limit overflow
  refine ⟨fun result => readDigits_spec limit overflow input 0 result (Nat.zero_le _), fun error => ?_⟩
  constructor
  · exact readDigits_error limit overflow error 0 input
  · rintro ⟨front, c, back, value, rfl, accepted, stopped⟩
    rw [readDigits_append, accepted]
    change readDigits limit overflow value (c :: back) = .error error
    by_cases valid : c.isDigit = true
    · obtain ⟨large, rfl⟩ : limit < appendDigit value c ∧ error = overflow := by simpa [valid] using stopped
      simp [readDigits, valid, Nat.not_le_of_gt large]
    · have same : error = .invalidDigit := by simpa [valid] using stopped
      simp [readDigits, valid, same]

private theorem integer_range (input : List Char) (number : Int)
    (parsed : parseInteger input = .ok number) : -2147483648 ≤ number ∧ number ≤ 2147483647 := by
  cases input with
  | nil => cases parsed
  | cons first digits =>
    simp only [parseInteger] at parsed
    split at parsed
    · split at parsed
      · cases parsed
      · cases read : readDigits 2147483648 .negativeOverflow 0 digits with
        | error => simp [read, Except.map] at parsed
        | ok magnitude =>
          have size := (readDigits_spec _ _ _ _ _ (by decide) read).1
          have equal : -(magnitude : Int) = number := by simpa [read, Except.map] using parsed
          omega
    · cases read : readDigits 2147483647 .positiveOverflow 0 (first :: digits) with
      | error => simp [read, Except.map] at parsed
      | ok magnitude =>
        have size := (readDigits_spec _ _ _ _ _ (by decide) read).1
        have equal : (magnitude : Int) = number := by simpa [read, Except.map] using parsed
        omega

theorem scoreContract_proven : ScoreContract := by
  intro input score result
  unfold parseScore at result
  cases parsed : parseInteger ((trim input).toList.dropWhile (· == '+')) with
  | error => simp [parsed, Except.mapError] at result
  | ok number =>
    have same : number = score := by simpa [parsed, Except.mapError] using result
    exact same ▸ integer_range _ _ parsed

private theorem rankFrom_range (square : Square.Square) (rank remaining : Nat)
    (scores : List MoveScore) (found : Nat) (answer : rankFrom square rank remaining scores = some found) :
    rank ≤ found ∧ found < rank + remaining := by
  induction remaining generalizing rank scores with
  | zero => simp [rankFrom] at answer
  | succ remaining ih =>
    cases scores with
    | nil => simp [rankFrom] at answer
    | cons first rest =>
      rw [rankFrom] at answer
      split at answer
      · cases answer
        omega
      · have bounds := ih _ _ answer
        omega

def scoreGroups : Nat → List MoveScore → List (List MoveScore)
  | 0, _ | _, [] => []
  | remaining + 1, first :: tail =>
    let equalScore := fun entry : MoveScore => entry.2 == first.2
    (first :: tail.takeWhile equalScore) :: scoreGroups remaining (tail.dropWhile equalScore)

private theorem rankFrom_groups (square : Square.Square) (rank remaining : Nat)
    (scores : List MoveScore) :
    rankFrom square rank remaining scores =
      Option.map (rank + ·)
        ((scoreGroups remaining scores).findIdx? (fun group => group.any (fun entry => entry.1 == square))) := by
  induction remaining generalizing rank scores with
  | zero => rfl
  | succ rem ih =>
    cases scores with
    | nil => rfl
    | cons first rest =>
      simp only [scoreGroups, rankFrom, List.findIdx?_cons]
      split
      · simp
      · rw [ih]
        simp [Option.map_map, Function.comp_def, Nat.add_right_comm, Nat.add_assoc]

def RankContract : Prop :=
  ∀ pos square,
    rankOf pos square =
      (scoreGroups 3 pos.moves).findIdx? (fun group => group.any (fun entry => entry.1 == square)) ∧
    (∀ rank, rankOf pos square = some rank → rank < 3)

theorem rankContract_proven : RankContract := by
  intro pos square
  refine ⟨?_, fun rank found => (rankFrom_range square 0 3 pos.moves rank found).2⟩
  simpa [rankOf] using rankFrom_groups square 0 3 pos.moves

private theorem ties_are_prefix (score : Int) (entries : List MoveScore)
    (sorted : entries.Pairwise (fun a b => moveOrder a b = true))
    (bounded : ∀ entry ∈ entries, entry.2 ≤ score) :
    entries.takeWhile (fun entry => entry.2 == score) =
      entries.filter (fun entry => entry.2 == score) := by
  induction entries with
  | nil => rfl
  | cons first tail ih =>
    by_cases tie : first.2 = score
    · simp [tie, ih sorted.tail (fun entry member => bounded entry (by simp [member]))]
    · have headBelow := bounded first (by simp)
      have tailBelow : ∀ entry ∈ tail, entry.2 < score := by
        intro entry member
        have ordered := (List.pairwise_cons.mp sorted).1 entry member
        have below : entry.2 ≤ first.2 := of_decide_eq_true ordered
        omega
      have empty : tail.filter (fun entry => entry.2 == score) = [] := by
        apply List.filter_eq_nil_iff.mpr
        intro entry member
        simpa using (Int.ne_of_lt (tailBelow entry member))
      simp [tie, empty]

def BestMoveContract : Prop :=
  ∀ pos : Position, pos.moves.Pairwise (fun a b => moveOrder a b = true) →
    match pos.moves with
    | [] => expectedScore pos = pos.passScore ∧ isPass pos = pos.passScore.isSome ∧ bestMoves pos = []
    | first :: rest =>
      expectedScore pos = some first.2 ∧ isPass pos = false ∧
        (∀ entry ∈ first :: rest, entry.2 ≤ first.2) ∧
        bestMoves pos = ((first :: rest).filter (fun entry => entry.2 == first.2)).map Prod.fst

theorem bestMoveContract_proven : BestMoveContract := by
  intro pos sorted
  cases entries : pos.moves with
  | nil => simp [expectedScore, isPass, bestMoves, entries]
  | cons first rest =>
    rw [entries] at sorted
    have bounded : ∀ entry ∈ first :: rest, entry.2 ≤ first.2 := by
      intro entry member
      rcases List.mem_cons.mp member with rfl | later
      · exact Int.le_refl _
      · exact of_decide_eq_true ((List.pairwise_cons.mp sorted).1 _ later)
    have same := ties_are_prefix first.2 (first :: rest) sorted bounded
    refine ⟨by simp [expectedScore, entries], by simp [isPass, entries], bounded, ?_⟩
    simpa [bestMoves, entries] using congrArg (List.map Prod.fst) same

def SuccessfulParseContract : Prop :=
  ∀ input pos, parse input = .ok (some pos) →
    pos.moves.Pairwise (fun a b => moveOrder a b = true)

theorem successfulParseContract_proven : SuccessfulParseContract := by
  intro input pos accepted
  simp only [parse] at accepted
  split at accepted
  · cases accepted
  · split at accepted
    · cases accepted
    · rename_i header segments parts
      cases parsed : parseHeader (trim header) with
      | error err => simp [parsed, Except.bind] at accepted
      | ok board =>
        cases all : segments.mapM parseSegment with
        | error err => simp [parsed, Except.bind, all, Except.map] at accepted
        | ok decoded =>
          obtain ⟨board, side⟩ := board
          simp [parsed, Except.bind, all, Except.map] at accepted
          rw [← accepted]
          exact (sortContract_proven _).2.1

def ScoreLookupContract : Prop :=
  ∀ (pos : Position) (square : Square.Square),
    pos.moves.Pairwise (fun a b => moveOrder a b = true) →
    match scoreOf pos square with
    | none => ∀ entry ∈ pos.moves, entry.1 ≠ square
    | some rating =>
      (square, rating) ∈ pos.moves ∧
        ∀ entry ∈ pos.moves, entry.1 = square → entry.2 ≤ rating

theorem scoreLookupContract_proven : ScoreLookupContract := by
  intro pos square sorted
  unfold scoreOf
  cases found : pos.moves.find? (fun entry => entry.1 == square) with
  | none =>
    intro entry member equal
    have missing := List.find?_eq_none.mp found entry member
    exact missing (by simp [equal])
  | some best =>
    obtain ⟨sameSquare, before, after, parts, earlier⟩ := List.find?_eq_some_iff_append.mp found
    have sq : best.1 = square := beq_iff_eq.mp sameSquare
    refine ⟨?_, ?_⟩
    · rw [← sq]
      change best ∈ pos.moves
      rw [parts]
      simp
    · intro entry member equal
      rw [parts] at member
      rcases List.mem_append.mp member with left | right
      · have missed := earlier entry left
        simp [equal] at missed
      · rcases List.mem_cons.mp right with rfl | later
        · exact Int.le_refl _
        · rw [parts] at sorted
          exact of_decide_eq_true
            ((List.pairwise_cons.mp ((List.pairwise_append.mp sorted).2.1)).1 _ later)

def HeaderContract : Prop :=
  ∀ header board side, parseHeader header = .ok (board, side) →
    let chars := header.toList
    (∀ char ∈ chars, char.toNat < 128) ∧ 65 ≤ chars.length ∧
    (trim (String.ofList (chars.drop 64)) = "X" ∧ side = .black ∨
      trim (String.ofList (chars.drop 64)) = "O" ∧ side = .white) ∧
    Board.fromString true (String.ofList (chars.take 64)) side = .ok board

theorem headerContract_proven : HeaderContract := by
  intro header board side accepted chars
  dsimp only [parseHeader] at accepted
  split at accepted
  · cases accepted
  · rename_i ascii
    have asciiChars : ∀ char ∈ chars, char.toNat < 128 := by
      intro char member
      have all : header.toList.all (fun c => decide (c.toNat < 128)) = true := by
        cases result : header.toList.all (fun c => decide (c.toNat < 128)) with
        | true => rfl
        | false => simp [result] at ascii
      exact of_decide_eq_true (List.all_eq_true.mp all char member)
    split at accepted
    · cases accepted
    · rename_i enough
      refine ⟨asciiChars, by exact Nat.le_of_not_gt enough, ?_⟩
      split at accepted
      next label =>
        cases decoded : Board.fromString true (String.ofList (chars.take 64)) .black with
        | error failure => simp [chars, decoded, Except.bind, Except.mapError, Except.map] at accepted
        | ok value =>
          simp [chars, decoded, Except.bind, Except.mapError, Except.map] at accepted
          obtain ⟨rfl, rfl⟩ := accepted
          exact ⟨.inl ⟨label, rfl⟩, decoded⟩
      next label =>
        cases decoded : Board.fromString true (String.ofList (chars.take 64)) .white with
        | error failure => simp [chars, decoded, Except.bind, Except.mapError, Except.map] at accepted
        | ok value =>
          simp [chars, decoded, Except.bind, Except.mapError, Except.map] at accepted
          obtain ⟨rfl, rfl⟩ := accepted
          exact ⟨.inr ⟨label, rfl⟩, decoded⟩
      · cases accepted
      · cases accepted

end NeuralReversi.Obf
