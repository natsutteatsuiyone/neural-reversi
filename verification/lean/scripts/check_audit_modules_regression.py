from pathlib import Path
import tempfile

from audit_modules import DRIVER_TEXT, validate

ALL_LIB_ROOTS = (
    'NeuralReversi',
    'NeuralReversi.AuditImports',
    'Audit',
    'DiffMain',
    'CSimpAuditRegression',
)


def render_lakefile(lib_roots: tuple[str, ...], exe_root: str = 'DiffMain') -> str:
    roots_toml = ', '.join(f'"{root}"' for root in lib_roots)
    return (
        'name = "auditFixture"\n'
        '[[lean_lib]]\n'
        'name = "NeuralReversi"\n'
        f'roots = [{roots_toml}]\n'
        '[[lean_exe]]\n'
        'name = "revmodel"\n'
        f'root = "{exe_root}"\n'
    )


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode('utf-8'))


def rejected(root: Path) -> None:
    try:
        validate(root)
    except (ValueError, OSError):
        return
    raise AssertionError('Incomplete audit metadata was accepted')


def main() -> None:
    positives = negatives = 0
    with tempfile.TemporaryDirectory(prefix='audit-modules-regression-') as directory:
        root = Path(directory)
        configuration = render_lakefile(ALL_LIB_ROOTS)
        write(root / 'lakefile.toml', configuration)
        for name in ('NeuralReversi', 'DiffMain', 'CSimpAuditRegression', 'NeuralReversi/Helper'):
            write(root / f'{name}.lean', '-- Valid inventory fixture; not executed.\n')
        write(root / 'Audit.lean', DRIVER_TEXT)
        validate(root, write=True)
        validate(root)
        positives += 2
        for missing in ('DiffMain', 'Audit'):
            write(
                root / 'lakefile.toml',
                render_lakefile(tuple(r for r in ALL_LIB_ROOTS if r != missing)),
            )
            rejected(root)
            negatives += 1
        write(root / 'lakefile.toml', configuration)
        validate(root)
        positives += 1
        orphan = root / 'NeuralReversi/Orphan.lean'
        write(orphan, '-- Orphan source must enter the complete audit.\n')
        rejected(root)
        negatives += 1
        validate(root, write=True)
        validate(root)
        positives += 2
        aggregate = root / 'NeuralReversi/AuditImports.lean'
        original = aggregate.read_text(encoding='utf-8')
        for value in (
            original.replace('import NeuralReversi.Helper\n', ''),
            original + 'import NeuralReversi.Helper\n',
        ):
            write(aggregate, value)
            rejected(root)
            negatives += 1
        write(aggregate, original)
        validate(root)
        positives += 1
        orphan.unlink()
        rejected(root)
        negatives += 1
        validate(root, write=True)
        validate(root)
        positives += 2
        write(root / 'Audit.lean', DRIVER_TEXT + '\ntheorem hiddenAfterAudit : True := True.intro\n')
        rejected(root)
        negatives += 1
        write(root / 'Audit.lean', DRIVER_TEXT)
        validate(root)
        positives += 1
        ambiguous = root / 'NeuralReversi/Helper.Bad.lean'
        write(ambiguous, '-- Ambiguous dotted path.\n')
        rejected(root)
        negatives += 1
        ambiguous.unlink()
        validate(root)
        positives += 1
        write(root / 'UnregisteredRegression.lean', '-- A top-level source needs a build root.\n')
        rejected(root)
        negatives += 1
        write(
            root / 'lakefile.toml',
            render_lakefile((*ALL_LIB_ROOTS, 'UnregisteredRegression')),
        )
        validate(root, write=True)
        validate(root)
        positives += 2
        inventory = root / 'NeuralReversi/AuditModules.lean'
        original = inventory.read_text(encoding='utf-8')
        write(inventory, original.replace('  ' + chr(96) + 'DiffMain,\n', ''))
        rejected(root)
        negatives += 1
        write(inventory, original)
        validate(root)
        positives += 1
    print(f'AUDIT MODULES REGRESSION OK ({positives} positive checks; {negatives} rejected controls)')


if __name__ == '__main__':
    main()
