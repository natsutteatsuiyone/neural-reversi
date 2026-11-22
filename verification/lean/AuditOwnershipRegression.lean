import NeuralReversi
import NeuralReversi.AuditPrivateFixture

open Lean Elab Command NeuralReversi.Audit

private theorem localAuditOwnershipProof : 7+7=14 := rfl

def auditUnprefixedSource (n : Nat) : Nat := n+0
def auditUnprefixedTarget (n : Nat) : Nat := n

theorem auditUnprefixedReplacement : auditUnprefixedSource = auditUnprefixedTarget := by
  funext n
  simp [auditUnprefixedSource,auditUnprefixedTarget]

#eval show CommandElabM Unit from do
  let env ← getEnv
  let baseline ← auditProjectCSimp env
  unless baseline == 2 do throwError "Ownership regression baseline differs"
  let fixtureModule := `NeuralReversi.AuditPrivateFixture
  unless declarationOwnedBy env [fixtureModule] `unprefixedImportedAuditProof do
    throwError "Unprefixed imported project proof was missed"
  unless declarationOwnedBy env [fixtureModule] `Nat.projectAuditProofInForeignNamespace do
    throwError "Foreign-namespace project proof was missed"
  if declarationOwnedBy env [fixtureModule] `NeuralReversi.Disc.colorExchange_proven then
    throwError "A namespace lookalike outside the configured module set was misowned"
  if declarationOwnedBy env [env.mainModule] `unknownAuditOwnershipDeclaration then
    throwError "Unknown declaration was misowned"
  let mut importedPrivate := false
  let mut currentPrivate := false
  for (name,info) in env.constants.toList do
    if info.isTheorem && (privateToUserName? name) == some `hiddenAuditOwnershipProof then
      unless declarationOwnedBy env [fixtureModule] name do
        throwError "Unused imported private proof was missed"
      if declarationOwnedBy env [fixtureModule] `hiddenAuditOwnershipProof then
        throwError "Demangling before provenance lookup concealed an unknown name"
      importedPrivate := true
    if info.isTheorem && (privateToUserName? name) == some `localAuditOwnershipProof then
      unless (env.getModuleIdxFor? name).isNone && declarationOwnedBy env [env.mainModule] name do
        throwError "Current private proof was missed"
      currentPrivate := true
  unless importedPrivate && currentPrivate do throwError "Private ownership controls were not loaded"
  let entry : Lean.Compiler.CSimp.Entry := {
    fromDeclName := `auditUnprefixedSource
    toDeclName := `auditUnprefixedTarget
    thmName := `auditUnprefixedReplacement }
  let altered := Lean.Compiler.CSimp.ext.modifyState env fun state =>
    {state with map := state.map.insert entry.fromDeclName entry}
  let rejected ← try
    let _ ← auditProjectCSimp altered
    pure false
  catch _ => pure true
  unless rejected do throwError "Unprefixed project compiler replacement escaped admission"
  let stillBaseline ← auditProjectCSimp env
  unless stillBaseline == baseline do throwError "Copied-environment control changed the baseline"
  logInfo "AUDIT OWNERSHIP REGRESSION OK (unprefixed, foreign namespace, imported/current private, unknown/demangled/foreign names, copied CSimp control)"
