module
public import NeuralReversi.PrincipalVariation.Buffer

@[expose] public section

namespace NeuralReversi.PrincipalVariation.StackAccess

def initial : Stack := List.replicate capacity (List.replicate capacity none)

def put (stack : Stack) (ply : Nat) (row : Buffer) : Option Stack :=
  stack[ply]?.map fun _ => stack.set ply row

def clearAt (stack : Stack) (ply : Nat) : Option Stack :=
  stack[ply]?.map fun row => stack.set ply (clear row)

def prepareAt (stack : Stack) (ply : Nat) : Option Stack :=
  stack[ply]?.map fun _ => prepare stack ply

def promote (stack : Stack) (ply : Nat) (sq : Bitboard.Square) : Option Stack := do
  let parent ← stack[ply]?
  let child ← stack[ply+1]?
  pure (stack.set ply (prepend sq child parent))

end NeuralReversi.PrincipalVariation.StackAccess
