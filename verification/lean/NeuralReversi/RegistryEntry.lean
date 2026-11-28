module
public import NeuralReversi.JsonLite

@[expose] public section

namespace NeuralReversi

structure Entry where
  id : String
  functions : List String
  status : String
  property : String
  theoremName : String
  notes : String := ""
  bug : String := ""

def Entry.toJsonCompact (e : Entry) : String :=
  "{" ++
    "\"bug\":" ++ JsonLite.quoteString e.bug ++ "," ++
    "\"functions\":" ++ JsonLite.renderStringList e.functions ++ "," ++
    "\"id\":" ++ JsonLite.quoteString e.id ++ "," ++
    "\"notes\":" ++ JsonLite.quoteString e.notes ++ "," ++
    "\"property\":" ++ JsonLite.quoteString e.property ++ "," ++
    "\"status\":" ++ JsonLite.quoteString e.status ++ "," ++
    "\"theoremName\":" ++ JsonLite.quoteString e.theoremName ++
  "}"

def entriesToJsonCompact (es : List Entry) : String :=
  "[" ++ String.intercalate "," (es.map Entry.toJsonCompact) ++ "]"

end NeuralReversi
