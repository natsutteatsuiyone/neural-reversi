import NeuralReversi

-- Repeating an imported, valid registration creates another staged record.
attribute [csimp] NeuralReversi.Simd.X86.ternaryWord_exec

open Lean Elab Command NeuralReversi.Audit

#eval show CommandElabM Unit from do
  let env ← getEnv
  let count ← auditProjectCSimp env
  unless count == 2 do throwError "Repeated valid registration changed the active count"
  let state := Lean.Compiler.CSimp.ext.getState env
  let source := `NeuralReversi.Simd.X86.ternaryWord
  let missing := `NeuralReversi.Simd.X86.ternarylogicEpi64
  let some entry := state.map.find? source | throwError "Missing baseline word mapping"
  let staged : SMap Name Lean.Compiler.CSimp.Entry := {
    stage₁ := false
    map₁ := ({} : Std.HashMap Name Lean.Compiler.CSimp.Entry).insert source entry
    map₂ := ({} : PHashMap Name Lean.Compiler.CSimp.Entry).insert source entry }
  let altered := Lean.Compiler.CSimp.ext.modifyState env fun s => { s with map := staged }
  unless (Lean.Compiler.CSimp.ext.getState altered).map.find? missing |>.isNone do
    throwError "Missing-mapping regression setup failed"
  let rejected ← try
    let _ ← auditProjectCSimp altered
    pure false
  catch _ => pure true
  unless rejected do throwError "Duplicate staged records concealed a missing replacement"
  logInfo "CSIMP REGRESSION OK (repeated valid registration; duplicated records with missing source)"
