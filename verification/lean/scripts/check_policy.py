import re
import sys
from pathlib import Path
from typing import NoReturn

LEAN_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(
    r"\b(sorry|sorryAx|admit|axiom|native_decide|bv_decide|ofReduceBool|"
    r"ofReduceNat|trustCompiler|implemented_by|extern|unsafe|partial)\b"
)
CHARACTER = re.compile(r"'(?:\\(?:[rnt\\\"']|x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4})|[^'\\])'")
RAW_STRING = re.compile(r'r(#+)?"')
ARGS_BEFORE_INTERPOLATED_MESSAGE = {
    "s!": 0, "m!": 0, "f!": 0, "throwError": 0, "throwErrorAt": 1,
    "throwNamedError": 1, "throwNamedErrorAt": 2,
    "logNamedError": 1, "logNamedErrorAt": 2, "dbg_trace": 0,
}
_INTERPOLATED_NAMES = sorted(ARGS_BEFORE_INTERPOLATED_MESSAGE, key=len, reverse=True)
_PREFIX_ALTERNATION = "|".join(re.escape(name) for name in _INTERPOLATED_NAMES)
INTERPOLATED_PREFIX = re.compile(r"(?:" + _PREFIX_ALTERNATION + r")(?![\w'!?])")
TRACE_PREFIX = re.compile(r"(?:Macro\.trace|trace)(?![\w'!?])")
_CODE_COMMON = (
    r"--|/-|«|r#*\x22|(?:" + _PREFIX_ALTERNATION + r"|Macro\.trace|trace)(?![\w'!?])|'|\x22"
)
CODE_TOP_TOKEN = re.compile(_CODE_COMMON)
CODE_GROUP_TOKEN = re.compile(_CODE_COMMON + r"|[()\[\]{}⟨⟩]")
BLOCK_COMMENT_TOKEN = re.compile(r"/-|-/")
STRING_PLAIN_TOKEN = re.compile(r'["\\]')
STRING_INTERP_TOKEN = re.compile(r'["\\{]')
WHITESPACE = re.compile(r"\s*")
GROUPS = {"(": ")", "[": "]", "{": "}", "⟨": "⟩"}


class PolicySyntaxError(ValueError):
    pass


def strip_comments_and_strings(text: str) -> str:
    parts: list[str] = []
    copied = 0
    size = len(text)

    def mask(start: int, end: int) -> None:
        nonlocal copied
        if start > copied:
            parts.append(text[copied:start])
        segment = text[start:end]
        if "\n" in segment:
            parts.append("\n".join(" " * len(line) for line in segment.split("\n")))
        else:
            parts.append(" " * (end - start))
        copied = end

    def fail(kind: str, start: int) -> NoReturn:
        line = text.count("\n", 0, start) + 1
        raise PolicySyntaxError(f"line {line}: unterminated {kind}")

    def comment(start: int) -> int:
        if text.startswith("--", start):
            end = text.find("\n", start)
            end = size if end == -1 else end
        else:
            end, depth = start + 2, 1
            while depth:
                match = BLOCK_COMMENT_TOKEN.search(text, end)
                if match is None:
                    fail("block comment", start)
                if match.group() == "/-":
                    depth += 1
                else:
                    depth -= 1
                end = match.end()
        mask(start, end)
        return end

    def trivia(start: int) -> int:
        pos = start
        while pos < size:
            ws = WHITESPACE.match(text, pos)
            if ws is not None and ws.end() > pos:
                pos = ws.end()
                continue
            if text.startswith(("--", "/-"), pos):
                pos = comment(pos)
            else:
                break
        return pos

    def identifier_char(char: str) -> bool:
        return char.isalnum() or char in "_'!?"

    def boundary(pos: int) -> bool:
        return pos == 0 or not identifier_char(text[pos - 1])

    def string(start: int, interpolated: bool) -> int:
        scanner = STRING_INTERP_TOKEN if interpolated else STRING_PLAIN_TOKEN
        seg_start = start
        pos = start + 1
        while pos < size:
            match = scanner.search(text, pos)
            if match is None:
                break
            pos = match.start()
            char = match.group()
            if char == "\\":
                pos = min(pos + 2, size)
            elif char == '"':
                mask(seg_start, pos + 1)
                return pos + 1
            else:
                mask(seg_start, pos)
                pos = code(pos + 1, closing="}", opening=pos)
                seg_start = pos
        fail("interpolated string" if interpolated else "string", start)

    def name_end(start: int) -> int:
        pos = start
        while pos < size:
            if text[pos] == "«":
                end = text.find("»", pos + 1)
                if end == -1:
                    fail("escaped identifier", pos)
                pos = end + 1
            else:
                end = pos
                while end < size and identifier_char(text[end]):
                    end += 1
                if end == pos:
                    return pos
                pos = end
            if pos + 1 < size and text[pos] == "." and (
                    identifier_char(text[pos + 1]) or text[pos + 1] == "«"):
                pos += 1
            else:
                break
        return pos

    def argument(start: int) -> int:
        pos = trivia(start)
        begin = pos
        while pos < size and text[pos] == "`":
            pos += 1
        if pos < size and text[pos] in GROUPS:
            pos = code(pos + 1, closing=GROUPS[text[pos]], opening=pos)
        elif pos < size and (text[pos].isalnum() or text[pos] in "_«"):
            pos = name_end(pos)
        else:
            line = text.count("\n", 0, begin) + 1
            raise PolicySyntaxError(
                f"line {line}: ambiguous interpolation prefix argument; "
                "parenthesize it or use a qualified function and explicit m!"
            )
        while pos < size:
            if text[pos] == "." and pos + 1 < size and (
                    identifier_char(text[pos + 1]) or text[pos + 1] == "«"):
                pos = name_end(pos + 1)
            elif text[pos] == "[":
                pos = code(pos + 1, closing="]", opening=pos)
            else:
                break
        return pos

    def skip_escaped_identifier_unmasked(start: int) -> int:
        end = text.find("»", start + 1)
        if end == -1:
            fail("escaped identifier", start)
        return end + 1

    def code(start: int, closing: str | None = None, opening: int = 0) -> int:
        scanner = CODE_TOP_TOKEN if closing is None else CODE_GROUP_TOKEN
        pos = start
        while pos < size:
            match = scanner.search(text, pos)
            if match is None:
                pos = size
                break
            pos = match.start()
            token = match.group()
            if token == "--" or token == "/-":
                pos = comment(pos)
            elif token == "«":
                pos = skip_escaped_identifier_unmasked(pos)
            elif token == '"':
                pos = string(pos, interpolated=False)
            elif token == "'":
                if boundary(pos):
                    char = CHARACTER.match(text, pos)
                    if char:
                        mask(pos, char.end())
                        pos = char.end()
                        continue
                pos += 1
            elif token[0] == "r" and token[-1] == '"':
                if boundary(pos):
                    delimiter = '"' + "#" * (len(token) - 2)
                    end = text.find(delimiter, match.end())
                    if end == -1:
                        fail("raw string", pos)
                    end += len(delimiter)
                    mask(pos, end)
                    pos = end
                else:
                    pos += 1
            elif token in ARGS_BEFORE_INTERPOLATED_MESSAGE:
                if boundary(pos) and (pos == 0 or text[pos - 1] != "."):
                    message = match.end()
                    for _ in range(ARGS_BEFORE_INTERPOLATED_MESSAGE[token]):
                        message = argument(message)
                    quote = trivia(message)
                    if quote < size and text[quote] == '"':
                        pos = string(quote, interpolated=True)
                    else:
                        pos = quote
                else:
                    pos += 1
            elif token == "trace" or token == "Macro.trace":
                if boundary(pos) and (pos == 0 or text[pos - 1] != "."):
                    bracket = trivia(match.end())
                    if bracket < size and text[bracket] == "[":
                        message = code(bracket + 1, closing="]", opening=bracket)
                        quote = trivia(message)
                        if quote < size and text[quote] == '"':
                            pos = string(quote, interpolated=True)
                        else:
                            pos = quote
                    else:
                        pos = bracket
                else:
                    pos += 1
            else:
                if token == closing:
                    return pos + 1
                if token in GROUPS:
                    pos = code(pos + 1, closing=GROUPS[token], opening=pos)
                else:
                    line = text.count("\n", 0, pos) + 1
                    raise PolicySyntaxError(f"line {line}: mismatched interpolation delimiter")
        if closing is not None:
            fail("interpolation expression or prefix argument", opening)
        return pos

    code(0)
    if copied == 0:
        return text
    if copied < size:
        parts.append(text[copied:])
    return "".join(parts)


def lean_files() -> list[tuple[Path, Path]]:
    files: list[tuple[Path, Path]] = [
        (path, path.relative_to(LEAN_ROOT))
        for path in sorted(LEAN_ROOT.rglob("*.lean"))
        if ".lake" not in path.relative_to(LEAN_ROOT).parts
    ]
    witness_root = LEAN_ROOT.parent / "tla" / "lean-witnesses"
    if witness_root.is_dir():
        files.extend(
            (path, path.relative_to(LEAN_ROOT.parent))
            for path in sorted(witness_root.rglob("*.lean"))
        )
    return files


def main() -> None:
    failures = []
    files = lean_files()
    for path, relative in files:
        try:
            code = strip_comments_and_strings(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, PolicySyntaxError) as error:
            failures.append(f"{relative}: {error}")
            continue
        for match in FORBIDDEN.finditer(code):
            line = code.count("\n", 0, match.start()) + 1
            failures.append(f"{relative}:{line}: {match.group()}")
    if failures:
        sys.exit("PROOF POLICY FAIL\n" + "\n".join(failures))
    print(f"PROOF POLICY OK ({len(files)} Lean files; kernel audit is also required)")


if __name__ == "__main__":
    main()
