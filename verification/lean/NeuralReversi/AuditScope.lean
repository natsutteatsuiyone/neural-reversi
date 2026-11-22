import NeuralReversi.AuditModules

namespace NeuralReversi.Audit
open Lean

def declarationOwnedBy (env : Environment) (modules : List Name) (name : Name) : Bool :=
  match env.getModuleIdxFor? name with
  | some index => modules.contains env.header.moduleNames[index.toNat]!
  | none => (env.find? name).isSome && modules.contains env.mainModule

def projectDeclaration (env : Environment) (name : Name) : Bool :=
  declarationOwnedBy env projectModules name

end NeuralReversi.Audit
