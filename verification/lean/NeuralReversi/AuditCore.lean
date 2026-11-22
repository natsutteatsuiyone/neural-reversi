import NeuralReversi.Registry

namespace NeuralReversi.Audit
open Lean Elab Command NeuralReversi

/-- Lean 4 generates a `._unsafe_rec` compiler auxiliary (with `isPartial = true`)
for every safe well-founded/structural recursive definition. A `partial` definition's
parent is an `opaqueInfo`, not a safe `defnInfo`, so it is still rejected. -/
private def isSafeCompilerUnsafeRec (env : Environment) (name : Name) (info : ConstantInfo) : Bool :=
  match name with
  | .str parent "_unsafe_rec" =>
    info.isPartial && !info.isUnsafe &&
    match env.find? parent with
    | some (.defnInfo v) => v.safety == .safe
    | _ => false
  | _ => false

/-- Lean 4 generates `T.ctorIdx._impl` (with `isUnsafe = true`) as the compiler
implementation of `T.ctorIdx` for every safe inductive type `T`. -/
private def isSafeCtorIdxImpl (env : Environment) (name : Name) (info : ConstantInfo) : Bool :=
  match name with
  | .str (.str indName "ctorIdx") "_impl" =>
    let ctorIdxName := .str indName "ctorIdx"
    info.isUnsafe && !info.isPartial &&
    Lean.Compiler.getImplementedBy? env ctorIdxName == some name &&
    match env.find? indName, env.find? ctorIdxName with
    | some (.inductInfo ind), some (.defnInfo ctorIdxDef) =>
      !ind.isUnsafe && ctorIdxDef.safety == .safe
    | _, _ => false
  | _ => false

private def isSafeCtorIdxImplementedBy (env : Environment) (name : Name) (implName : Name) : Bool :=
  match name with
  | .str indName "ctorIdx" =>
    implName == .str name "_impl" &&
    match env.find? indName, env.find? name with
    | some (.inductInfo ind), some (.defnInfo ctorIdxDef) =>
      !ind.isUnsafe && ctorIdxDef.safety == .safe
    | _, _ => false
  | _ => false

def auditRegistryEntries (env : Environment) (registryEntries : List Entry) : CommandElabM Nat := do
  let mut seen : List String := []
  let mut settled : Nat := 0
  for entry in registryEntries do
    if entry.id ∈ seen then throwError "duplicate property id: {entry.id}"
    seen := entry.id :: seen
    unless entry.status ∈ ["proven", "refuted"] do
      throwError "{entry.id}: invalid or open status {entry.status}"
    let propName := entry.property.toName
    let some propInfo := env.find? propName
      | throwError "{entry.id}: unknown property {propName}"
    unless propInfo.type == mkSort .zero do
      throwError "{entry.id}: property must be a closed Prop"
    let thmName := entry.theoremName.toName
    unless projectDeclaration env thmName do
      throwError "{entry.id}: {thmName} is not a project declaration"
    let some (.thmInfo thm) := env.find? thmName
      | throwError "{entry.id}: {thmName} must be a theorem"
    let expected := if entry.status == "refuted" then
      mkApp (mkConst ``Not) (mkConst propName) else mkConst propName
    unless thm.type == expected do
      throwError "{entry.id}: theorem type does not match registered property"
    if entry.status == "refuted" && entry.bug.isEmpty then
      throwError "{entry.id}: refutation needs a local bug record"
    settled := settled + 1
  return settled

def auditProjectDeclarations (modules : List Name := projectModules) : CommandElabM (Nat × Nat) := do
  let env ← getEnv
  let mut declCount : Nat := 0
  let mut theoremCount : Nat := 0
  for (name, info) in env.constants.toList do
    unless declarationOwnedBy env modules name do continue
    declCount := declCount + 1
    if info.isAxiom then throwError "project axiom forbidden: {name}"
    if info.isUnsafe || info.isPartial then
      unless isSafeCompilerUnsafeRec env name info || isSafeCtorIdxImpl env name info do
        throwError "project declaration must be safe: {name}"
    if let some implName := Lean.Compiler.getImplementedBy? env name then
      unless isSafeCtorIdxImplementedBy env name implName do
        throwError "project implemented_by attribute forbidden: {name} -> {implName}"
    if (Lean.getExternAttrData? env name).isSome then
      throwError "project extern attribute forbidden: {name}"
    if info.isTheorem then
      theoremCount := theoremCount + 1
    let axioms ← liftCoreM (collectAxioms name)
    for axiomName in axioms do
      unless axiomName ∈ [``propext, ``Classical.choice, ``Quot.sound] do
        throwError "{name}: unapproved axiom {axiomName}"
  return (theoremCount, declCount)

def runProjectAudit : CommandElabM Unit := do
  let env ← getEnv
  for moduleName in projectModules do
    if moduleName == env.mainModule then continue
    unless env.header.moduleNames.contains moduleName do
      throwError "Project module was not imported for audit: {moduleName}"
  let csimpCount ← NeuralReversi.Audit.auditProjectCSimp env
  let settled ← auditRegistryEntries env entries
  let (theoremCount, declCount) ← auditProjectDeclarations
  logInfo m!"AUDIT {entries.length} entries, {settled} settled, {theoremCount} theorems, {declCount} declarations checked"
  logInfo m!"CSIMP AUDIT {csimpCount} active project mappings checked"

end NeuralReversi.Audit
