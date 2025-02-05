import re

from rich.console import Console

console = Console()


def indent_code(code: str, nb_spaces: int = 2) -> str:
    return "\n".join(" " * nb_spaces + line for line in code.split("\n"))


def lean_comments_ranges(
    lean_code: str, multiline_comment_suffix: str = "", remove_single_line_comments: bool = True
) -> list[tuple[int, int]]:
    """Extract the ranges of Lean comments from a Lean code snippet."""
    # TODO: this method does not handle strings and other potential edge cases (i.e. this method will probably crash if `/-`, `-/` or `--` are used in a string)

    # multiline comments
    open_comment_indices = [m.start() for m in re.finditer(r"/-" + multiline_comment_suffix, lean_code)]
    close_comment_indices = [
        m.start() + len(multiline_comment_suffix) + 2 for m in re.finditer(multiline_comment_suffix + r"-/", lean_code)
    ]

    if len(open_comment_indices) == len(close_comment_indices) + 1:
        # the last comment has probably not been closed due to partial code
        close_comment_indices.append(len(lean_code))

    elif len(open_comment_indices) + 1 == len(close_comment_indices):
        # the first comment has probably been opened before the code snippet
        open_comment_indices.insert(0, 0)

    elif len(open_comment_indices) != len(close_comment_indices):
        raise ValueError("Mismatched open and close comment indices.")

    # trick to handle nested comments in a simple way
    multiline_comment_ranges = list(zip(open_comment_indices, close_comment_indices))

    if remove_single_line_comments:
        # single line comments
        single_line_comment_ranges = [
            (m.start(), lean_code.find("\n", m.start())) for m in re.finditer(r"--", lean_code)
        ]
        multiline_comment_ranges += single_line_comment_ranges

    # merge potential overlapping ranges
    comment_ranges = sorted(multiline_comment_ranges, key=lambda x: x[0])
    merged_comment_ranges = []
    for start, end in comment_ranges:
        if merged_comment_ranges and start <= merged_comment_ranges[-1][1]:
            merged_comment_ranges[-1] = (merged_comment_ranges[-1][0], max(merged_comment_ranges[-1][1], end))
        else:
            merged_comment_ranges.append((start, end))

    return merged_comment_ranges


def remove_lean_comments(lean_code: str) -> str | None:
    try:
        comment_ranges = lean_comments_ranges(lean_code)

        new_lean_code = ""
        prev_start = 0
        for start, end in comment_ranges:
            new_lean_code += lean_code[prev_start:start]
            prev_start = end

        new_lean_code += lean_code[prev_start:]
        return new_lean_code

    except Exception:
        return None


def split_implementation(declaration: str, start: int = 0):
    # for a theorem, an implementation is the proof
    if ":=" in declaration:
        # we have to be careful here as ":=" can be used inside the declaration itself
        indices = set([m.start() for m in re.finditer(r":=", declaration)])

        # we remove the ones related to "let", "haveI", ... declarations
        for keyword in ["let", "haveI"]:
            regex = rf"{keyword}\s+\S*?\s*(:=)"
            decl_indices = set([m.start(1) for m in re.finditer(regex, declaration)])
            indices = indices - decl_indices

        # implementation using pcre2 blows up the memory, and it turns out it is faster to use a python loop
        counters = {"(": 0, "{": 0, "[": 0}
        closing = {")": "(", "}": "{", "]": "["}
        for i, c in enumerate(declaration[start:]):
            if c in counters:
                counters[c] += 1
            elif c in [")", "}", "]"]:
                counters[closing[c]] -= 1
            if all([v == 0 for v in counters.values()]) and (i + start) in indices:
                return i + start + 2

    # TODO: handle other cases where the implementation starts with something else than ":="
    return None


def split_conclusion(declaration: str, start: int = 0) -> int | None:
    counters = {"(": 0, "{": 0, "[": 0}
    closing = {")": "(", "}": "{", "]": "["}
    for i, c in enumerate(declaration[start:]):
        if c in counters:
            counters[c] += 1
        elif c in [")", "}", "]"]:
            counters[closing[c]] -= 1
        if all([v == 0 for v in counters.values()]) and c == ":":
            return i + start
    return None


def clean_theorem_string(theorem_string: str, new_theorem_name: str = "dummy", add_sorry: bool = True) -> str | None:
    """Clean a theorem string by removing the proof, comments, and updating the theorem name.
    This method assumes that no other declarations are present in the theorem string."""
    try:
        # clean the theorem string
        clean_formal = remove_lean_comments(theorem_string)
        if clean_formal is None:
            raise ValueError("Comment removal failed.")
        clean_formal = re.sub(r"\s+", " ", clean_formal).strip()

        # we remove the first part of the string until the first "theorem" or "lemma" keyword
        theorem_decl_keywords = "|".join(["theorem", "lemma"])
        re_match = re.search(rf"\b{theorem_decl_keywords}\s", clean_formal)
        if re_match is None:
            raise ValueError("Theorem declaration keyword not found.")
        idx_theorem = re_match.start()
        clean_formal = clean_formal[idx_theorem:]

        # if a proof is provided we remove it
        idx_implement = split_implementation(clean_formal)
        if idx_implement is not None:
            clean_formal = clean_formal[:idx_implement].strip()

        if not clean_formal.endswith(":="):
            raise ValueError("Partial theorem declaration found.")

        # remove "theorem" and the theorem name
        clean_formal = re.sub(r"^[^\s]+", "", clean_formal).strip()
        clean_formal = re.sub(r"^[^\s:({\[]+", "", clean_formal).strip()
        clean_formal = f"theorem {new_theorem_name} " + clean_formal

        if add_sorry:
            clean_formal += " sorry"
        return clean_formal
    except Exception:
        return None


def extract_last_theorem(lean_code: str) -> int:
    """Extract the last theorem from a Lean code snippet. It assumes that the Lean code snippet ends with a theorem."""
    comments_ranges = lean_comments_ranges(lean_code)

    # find last theorem by looking for `theorem` keyword surrounded by whitespaces, or by being at the beginning of the string
    theorem_decl_keywords = ["theorem", "lemma"]
    theorem_indices = []
    for keyword in theorem_decl_keywords:
        theorem_indices += [m.start() for m in re.finditer(rf"\b{keyword}\s", lean_code)]

    # remove matches that are inside comments
    theorem_indices = [idx for idx in theorem_indices if not any(start <= idx <= end for start, end in comments_ranges)]

    if not theorem_indices:
        raise ValueError(f"No theorem found in the provided Lean code:\n{lean_code}")

    return theorem_indices[-1]


def clean_last_theorem_string(lean_code: str, new_theorem_name: str = "dummy", add_sorry: bool = False) -> str:
    """Clean the last theorem string from a Lean code snippet. It assumes that the Lean code snippet ends with a theorem."""
    idx_last_theorem = extract_last_theorem(lean_code)
    clean_thm = clean_theorem_string(lean_code[idx_last_theorem:], new_theorem_name, add_sorry=add_sorry)
    if clean_thm is not None:
        return lean_code[:idx_last_theorem] + clean_thm

    raise ValueError(f"Theorem extraction failed for the following Lean code:\n{lean_code}")


def message_intersects_code(message, start_line, end_line):
    res = True
    if start_line is not None:
        if message["endPos"]:
            res = res and message["endPos"]["line"] >= start_line
    if end_line is not None:
        if message["startPos"]:
            res = res and message["startPos"]["line"] <= end_line
    return res


def extract_exact_proof(
    lean_output: dict, proof_start_line: int | None = None, proof_end_line: int | None = None, verbose: bool = False
) -> str | None:
    # check only the messages intersecting the proof
    for message in lean_output.get("messages", []):
        if message_intersects_code(message, proof_start_line, proof_end_line):
            if message["severity"] == "error":
                return None
            if message["severity"] == "info" and message["data"].startswith("Try this:"):
                return message["data"].split("Try this:")[1].strip()
    return None


def is_valid_lean(
    lean_output: dict, start_line: int | None = None, end_line: int | None = None, allow_sorry: bool = True
):
    # check only the messages intersecting the code
    errors = [
        message
        for message in lean_output.get("messages", [])
        if message_intersects_code(message, start_line, end_line) and message["severity"] == "error"
    ]
    sorries = [
        message for message in lean_output.get("sorries", []) if message_intersects_code(message, start_line, end_line)
    ]
    return not errors and (allow_sorry or not sorries) and "env" in lean_output
