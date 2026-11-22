import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import check_policy as policy


class PolicyRegression(unittest.TestCase):
    def tokens(self, source: str) -> list[str]:
        masked = policy.strip_comments_and_strings(source)
        self.assertEqual(len(source), len(masked))
        self.assertEqual([i for i, c in enumerate(source) if c == "\n"],
                         [i for i, c in enumerate(masked) if c == "\n"])
        return policy.FORBIDDEN.findall(masked)

    def test_issue_reproductions(self) -> None:
        for source in [
            "def quote : Char := '\"'\ndef hidden : Nat := by sorry\n",
            'def hidden : String := s!"{(by sorry : Nat)}"\n',
            'def text : String := r#"a"b"#\ndef hidden : Nat := by sorry\n',
        ]:
            with self.subTest(source=source):
                self.assertEqual(self.tokens(source), ["sorry"])

    def test_inert_text(self) -> None:
        for source in [
            '-- sorry\ndef x := 0',
            '/- sorry /- axiom -/ admit -/\ndef x := 0',
            'def x := "sorry axiom native_decide"',
            r'def x := "\"sorry\" \\ admit"',
            'def x := r#"sorry "axiom" admit"#',
            'def x := r##"sorry "# axiom "##',
            'def x := r"sorry \\"',
            'def x := s!"sorry {1 + 2} axiom"',
            r'def x := s!"\{sorry} {1} axiom"',
            'def x := m!"unapproved axiom {name}"',
            'def x := s!"{("sorry" ++ r#"axiom"#)}"',
            'def x := s!"{(1 /- sorry -/)}"',
            'def x := panic! "{sorry}"',
            'def x := Lean.throwError "{sorry}"',
        ]:
            with self.subTest(source=source):
                self.assertEqual(self.tokens(source), [])

    def test_characters_and_primed_names(self) -> None:
        for literal in ['\'"\'', "'{'", "'}'", "'«'", "'λ'", "'\n'",
                        r"'\''", r"'\\'", r"'\"'", r"'\x22'", r"'\u0022'"]:
            with self.subTest(literal=literal):
                self.assertEqual(self.tokens(f"def x := {literal}\ndef y := by sorry"), ["sorry"])
        self.assertEqual(self.tokens("def x' := 1\ndef x'' := x'\ndef y := by sorry"), ["sorry"])
        self.assertEqual(self.tokens("def x'a' := 1\ndef y := by sorry"), ["sorry"])

    def test_interpolation_expressions(self) -> None:
        for source in [
            'def x := m!"{(by sorry : Nat)}"',
            'def x := throwError "{(by sorry : Nat)}"',
            'def x := throwError /- comment -/ "{(by sorry : Nat)}"',
            'def x := s! "{(by sorry : Nat)}"',
            'def x := s! /- comment -/ "{(by sorry : Nat)}"',
            'def x := s! -- comment\n"{(by sorry : Nat)}"',
            'def x := s!"{s!"{(by sorry : Nat)}"}"',
            'def x := s!"{{ field := (by sorry : Nat) }}"',
            'def x := s!"{let c := \'}\'; (by sorry : Nat)}"',
            'def x := s!"{let c := \'"\'; (by sorry : Nat)}"',
            'def x := s!"{let s := r#"}"#; (by sorry : Nat)}"',
            'def x := s!"{/- } -/ (by sorry : Nat)}"',
            'def x := s!"{-- }\n(by sorry : Nat)}"',
        ]:
            with self.subTest(source=source):
                self.assertEqual(self.tokens(source), ["sorry"])

    def test_implicit_interpolation_prefix_arguments(self) -> None:
        for prefix in [
            "throwErrorAt ref", "throwErrorAt Syntax.missing",
            "throwErrorAt (← getRef)", "throwErrorAt (id ref).raw",
            "throwErrorAt ref[0]", "throwErrorAt «ref with spaces»",
            "throwErrorAt /- ref -/ (id (ref)) /- message -/",
            "throwNamedError lean.unknownIdentifier",
            "throwNamedErrorAt (← getRef) lean.unknownIdentifier",
            "logNamedError lean.unknownIdentifier",
            "logNamedErrorAt ref lean.unknownIdentifier",
            "dbg_trace", "trace[Meta.debug]", "Macro.trace[debug]", "f!",
        ]:
            with self.subTest(prefix=prefix):
                self.assertEqual(self.tokens(prefix + ' "{(by sorry : Nat)}"'), ["sorry"])
                self.assertEqual(self.tokens(prefix + ' "sorry {(1 : Nat)} axiom"'), [])
                self.assertEqual(self.tokens(prefix + ' m!"{(by sorry : Nat)}"'), ["sorry"])

    def test_prefix_arguments_are_not_erased(self) -> None:
        source = 'throwErrorAt (by sorry : Lean.Syntax) "{(by admit : Nat)}"'
        self.assertEqual(self.tokens(source), ["sorry", "admit"])
        source = 'throwErrorAt ref[(by sorry : Nat)] "ok"'
        self.assertEqual(self.tokens(source), ["sorry"])
        source = 'throwErrorAt (mkStrLit s!"{(by sorry : Nat)}") "ok"'
        self.assertEqual(self.tokens(source), ["sorry"])
        source = 'throwErrorAt (mkStrLit r#"quote \" and }"#) "{(by sorry : Nat)}"'
        self.assertEqual(self.tokens(source), ["sorry"])

    def test_qualified_functions_have_ordinary_strings(self) -> None:
        for function in ["Lean.throwErrorAt", "Lean.throwNamedError",
                         "Lean.throwNamedErrorAt", "Lean.logNamedError"]:
            with self.subTest(function=function):
                self.assertEqual(self.tokens(function + ' ref "{sorry}"'), [])
                self.assertEqual(self.tokens(function + ' ref m!"{by sorry}"'), ["sorry"])
        self.assertEqual(self.tokens('mythrowErrorAt ref "{sorry}"'), [])
        self.assertEqual(self.tokens('throwErrorAtHelper ref "{sorry}"'), [])
        self.assertEqual(self.tokens('«throwErrorAt» ref "{sorry}"'), [])

    def test_nested_groups_and_literals(self) -> None:
        for inner in [
            '(let x := [1, 2]; (by sorry : Nat))',
            '(let x := (1, 2); (by sorry : Nat))',
            '(let x := "(}])"; (by sorry : Nat))',
            '(let x := r#"(}])"#; (by sorry : Nat))',
            '(let x := ⟨0, by decide⟩; (by sorry : Nat))',
            '(let x := «}])»; (by sorry : Nat))',
        ]:
            with self.subTest(inner=inner):
                self.assertEqual(self.tokens('s!"{' + inner + '}"'), ["sorry"])

    def test_implicit_interpolation_diagnostics(self) -> None:
        for source in ['throwErrorAt (ref', 'throwErrorAt (ref] "ok"',
                       'throwErrorAt $ref "ok"', 'trace[debug "text"',
                       'throwErrorAt ref "{(by sorry : Nat)"']:
            with self.subTest(source=source):
                with self.assertRaises(policy.PolicySyntaxError):
                    policy.strip_comments_and_strings(source)
        source = 'throwErrorAt\n (id ref)\n /- note -/ "{(by sorry : Nat)}"'
        masked = policy.strip_comments_and_strings(source)
        match = policy.FORBIDDEN.search(masked)
        self.assertIsNotNone(match)
        self.assertEqual(masked.count("\n", 0, match.start()) + 1, 3)

    def test_ordinary_text_does_not_activate_macro_rules(self) -> None:
        for literal in ['"throwErrorAt ref {sorry}"', 'r#"throwErrorAt ref {sorry}"#',
                        '"trace[debug] {sorry}"', '"f! {sorry}"']:
            self.assertEqual(self.tokens('def text := ' + literal), [])
        self.assertEqual(self.tokens('/- throwErrorAt $ref "{sorry}" -/'), [])

    def test_comments_do_not_join_tokens(self) -> None:
        self.assertEqual(self.tokens("by/--/sorry"), ["sorry"])
        self.assertEqual(self.tokens("by/- /--/ -/sorry"), ["sorry"])
        self.assertEqual(self.tokens("by/-\n-/sorry"), ["sorry"])

    def test_escaped_identifiers_do_not_hide_following_code(self) -> None:
        for name in ['«--»', '«/-»', '«quote"»', '«{»']:
            with self.subTest(name=name):
                self.assertEqual(self.tokens(f"def {name} := 0\ndef x := by sorry"), ["sorry"])
        self.assertEqual(self.tokens("def «sorry» := 0"), ["sorry"])

    def test_all_forbidden_names(self) -> None:
        names = ["sorry", "sorryAx", "admit", "axiom", "native_decide", "bv_decide",
                 "ofReduceBool", "ofReduceNat", "trustCompiler", "implemented_by",
                 "extern", "unsafe", "partial"]
        for name in names:
            with self.subTest(name=name):
                self.assertEqual(self.tokens(f'def x := s!"{{{name}}}"'), [name])
                self.assertEqual(self.tokens(f"def x := Lean.{name}"), [name])

    def test_unsafe_and_partial_declarations(self) -> None:
        self.assertEqual(self.tokens("unsafe def bad : Nat := 0\n"), ["unsafe"])
        self.assertEqual(self.tokens("partial def loop (n : Nat) : Nat := loop n\n"), ["partial"])
        self.assertEqual(
            self.tokens(
                "-- unsafe partial\n"
                'def msg := "unsafe partial"\n'
                "def partial_sum := 0\n"
                "def unsafe_flag := 0\n"
            ),
            [],
        )

    def test_unterminated_constructs_fail_closed(self) -> None:
        for source in ['/- comment', '"text', 'r#"text', 'r##"text"#',
                       's!"{(1)', 's!"text', '«name', 's!"{/- comment']:
            with self.subTest(source=source):
                with self.assertRaises(policy.PolicySyntaxError):
                    policy.strip_comments_and_strings(source)

    def test_line_numbers(self) -> None:
        source = '/- first\n/- second -/\n-/\ndef c := \'"\'\ndef x := s!"{by sorry}"\n'
        masked = policy.strip_comments_and_strings(source)
        match = policy.FORBIDDEN.search(masked)
        self.assertIsNotNone(match)
        self.assertEqual(masked.count("\n", 0, match.start()) + 1, 5)

    def test_main_reports_both_policy_and_lexical_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lean_dir = Path(directory)
            (lean_dir / ".lake").mkdir(parents=True)
            (lean_dir / ".lake/ignored.lean").write_text("sorry", encoding="utf-8")
            (lean_dir / "A.lean").write_text("def c := '\"'\ndef x := by sorry\n", encoding="utf-8")
            (lean_dir / "B.lean").write_text('/- unfinished', encoding="utf-8")
            with patch.object(policy, "LEAN_ROOT", lean_dir), self.assertRaises(SystemExit) as caught:
                policy.main()
            message = str(caught.exception)
            self.assertIn("A.lean:2: sorry", message)
            self.assertIn("B.lean: line 1: unterminated block comment", message)
            self.assertNotIn("ignored.lean", message)
            (lean_dir / "A.lean").write_text('def x := "sorry"', encoding="utf-8")
            (lean_dir / "B.lean").unlink()
            output = io.StringIO()
            with patch.object(policy, "LEAN_ROOT", lean_dir), contextlib.redirect_stdout(output):
                policy.main()
            self.assertIn("PROOF POLICY OK (1 Lean files", output.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
