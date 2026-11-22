module
public import Lean

private theorem hiddenAuditOwnershipProof : True := True.intro

public theorem unprefixedImportedAuditProof : 2+2=4 := rfl

namespace Nat
public theorem projectAuditProofInForeignNamespace : 3+3=6 := rfl
end Nat
