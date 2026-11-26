import Lean
import Lean.Compiler.CSimpAttr
import NeuralReversi.AuditScope

namespace NeuralReversi.Audit
open Lean Elab Command

def expectedProjectCSimp : List (Name × Name × Name) := [
  (`NeuralReversi.Flip.Scalar.shiftCount32,
   `NeuralReversi.Flip.Scalar.shiftCount32Fast,
   `NeuralReversi.Flip.Scalar.shiftCount32_exec),
  (`NeuralReversi.Simd.X86.ternaryWord,
   `NeuralReversi.Simd.X86.ternaryWordFast,
   `NeuralReversi.Simd.X86.ternaryWord_exec),
  (`NeuralReversi.Simd.X86.ternarylogicEpi64,
   `NeuralReversi.Simd.X86.ternarylogicEpi64Fast,
   `NeuralReversi.Simd.X86.ternarylogicEpi64_exec)]

def auditProjectCSimp (env : Environment) : CommandElabM Nat := do
  let state := Lean.Compiler.CSimp.ext.getState env
  let mut count := 0
  let mut seen : List Name := []
  -- SMap.toList contains both stages, including shadowed duplicate keys.
  for (key, _) in state.map.toList do
    if key ∈ seen then continue
    seen := key :: seen
    let some entry := state.map.find? key | continue
    unless projectDeclaration env key || projectDeclaration env entry.fromDeclName || projectDeclaration env entry.toDeclName ||
        projectDeclaration env entry.thmName do continue
    unless key == entry.fromDeclName do throwError "CSIMP map key differs from source"
    unless expectedProjectCSimp.contains (entry.fromDeclName,entry.toDeclName,entry.thmName) do
      throwError "Unexpected active project CSIMP mapping: {entry.thmName}"
    let some (.thmInfo theoremInfo) := env.find? entry.thmName |
      throwError "CSIMP {entry.thmName} is not an ordinary theorem"
    for name in [entry.fromDeclName,entry.toDeclName] do
      let some (.defnInfo info) := env.find? name |
        throwError "CSIMP {name} is not a definition"
      unless info.safety == .safe do throwError "CSIMP {name} is unsafe or partial"
    let some (_, sourceExpr, targetExpr) := theoremInfo.type.eq? |
      throwError "CSIMP theorem is not equality"
    unless sourceExpr.isConstOf entry.fromDeclName && targetExpr.isConstOf entry.toDeclName do
      throwError "CSIMP mapping and theorem type differ"
    unless theoremInfo.levelParams.isEmpty do throwError "Unexpected CSIMP universe parameters"
    unless (state.map.find? entry.toDeclName).isNone do
      throwError "CSIMP target has another active replacement"
    let axioms ← liftCoreM (collectAxioms entry.thmName)
    for name in axioms do
      unless name ∈ [``propext,``Classical.choice,``Quot.sound] do
        throwError "CSIMP {entry.thmName} has an unapproved axiom"
    count := count+1
  for (sourceName,targetName,theoremName) in expectedProjectCSimp do
    let some entry := state.map.find? sourceName |
      throwError "Missing expected project CSIMP mapping: {sourceName}"
    unless entry.fromDeclName == sourceName && entry.toDeclName == targetName &&
        entry.thmName == theoremName do
      throwError "Expected project CSIMP mapping differs: {sourceName}"
  unless count == expectedProjectCSimp.length do throwError "Missing expected project CSIMP mapping"
  return count

end NeuralReversi.Audit
