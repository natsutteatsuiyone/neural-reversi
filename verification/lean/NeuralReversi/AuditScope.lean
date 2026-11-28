import NeuralReversi.AuditModules

namespace NeuralReversi.Audit
open Lean

structure OwnershipIndex where
  ownedModules : Array Bool
  ownsMain : Bool

def buildOwnershipIndex (env : Environment) (modules : List Name) : OwnershipIndex :=
  let modSet : Std.HashSet Name := modules.foldl (fun acc m => acc.insert m) {}
  { ownedModules := env.header.moduleNames.map modSet.contains
    ownsMain := modSet.contains env.mainModule }

def OwnershipIndex.owns (idx : OwnershipIndex) (env : Environment) (name : Name) : Bool :=
  match env.getModuleIdxFor? name with
  | some modIdx => idx.ownedModules[modIdx.toNat]?.getD false
  | none => idx.ownsMain && (env.find? name).isSome

def declarationOwnedBy (env : Environment) (modules : List Name) (name : Name) : Bool :=
  match env.getModuleIdxFor? name with
  | some index => modules.contains env.header.moduleNames[index.toNat]!
  | none => (env.find? name).isSome && modules.contains env.mainModule

def projectDeclaration (env : Environment) (name : Name) : Bool :=
  declarationOwnedBy env projectModules name

end NeuralReversi.Audit
