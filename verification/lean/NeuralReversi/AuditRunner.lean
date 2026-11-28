import NeuralReversi.AuditCore
import NeuralReversi.RegistryData

namespace NeuralReversi.Audit
open Lean Elab Command NeuralReversi

def runProjectAudit : CommandElabM Unit := do
  let env ← getEnv
  let importedModules : Std.HashSet Name :=
    env.header.moduleNames.foldl (fun acc moduleName => acc.insert moduleName) {}
  for moduleName in projectModules do
    if moduleName == env.mainModule then continue
    unless importedModules.contains moduleName do
      throwError "Project module was not imported for audit: {moduleName}"
  let csimpCount ← NeuralReversi.Audit.auditProjectCSimp env
  let settled ← auditRegistryEntries env entries
  let (theoremCount, declCount) ← auditProjectDeclarations
  logInfo m!"AUDIT {entries.length} entries, {settled} settled, {theoremCount} theorems, {declCount} declarations checked"
  logInfo m!"CSIMP AUDIT {csimpCount} active project mappings checked"

end NeuralReversi.Audit
