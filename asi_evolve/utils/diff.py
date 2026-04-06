"""Helpers for parsing and applying search/replace diffs."""

import re
from typing import List, Optional, Tuple


def extract_diffs(
    diff_text: str,
    diff_pattern: str = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
) -> List[Tuple[str, str]]:
    """
    Extract all diff blocks from an LLM response.

    Args:
        diff_text: Raw response text.
        diff_pattern: Regex pattern for the diff format.

    Returns:
        List of `(search_text, replace_text)` tuples.
    """
    diff_blocks = re.findall(diff_pattern, diff_text, re.DOTALL)
    return [(match[0].rstrip(), match[1].rstrip()) for match in diff_blocks]


def apply_diff(
    original_code: str,
    diff_text: str,
    diff_pattern: str = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE",
) -> str:
    """
    Apply search/replace diffs to a source string.

    Args:
        original_code: Original source text.
        diff_text: Raw diff text.
        diff_pattern: Regex pattern for the diff format.

    Returns:
        Updated source text.

    Raises:
        ValueError: If the diff cannot be applied.
    """
    diff_blocks = extract_diffs(diff_text, diff_pattern)

    if not diff_blocks:
        raise ValueError("No diff blocks found in the input text")

    result_code = original_code

    for i, (search_text, replace_text) in enumerate(diff_blocks):
        if search_text not in result_code:
            raise ValueError(
                f"Diff block {i+1}: Search text not found in code.\n"
                f"Search text:\n{search_text[:200]}..."
            )

        result_code = result_code.replace(search_text, replace_text, 1)

    return result_code


def apply_diff_blocks(
    original_code: str,
    diff_blocks: List[Tuple[str, str]],
) -> Tuple[str, int]:
    """
    Apply pre-parsed diff blocks to the original code.

    Args:
        original_code: Original source text.
        diff_blocks: Parsed diff blocks.

    Returns:
        Tuple of `(updated_code, applied_count)`.
    """
    result_code = original_code
    applied_count = 0

    for search_text, replace_text in diff_blocks:
        if search_text in result_code:
            result_code = result_code.replace(search_text, replace_text, 1)
            applied_count += 1

    return result_code, applied_count


def parse_full_rewrite(llm_response: str, language: str = "python") -> Optional[str]:
    """
    Extract a full code rewrite from an LLM response.

    Args:
        llm_response: Raw LLM response.
        language: Preferred fenced-code language tag.

    Returns:
        Extracted code, or `None` if no code block is found.
    """
    code_block_pattern = rf"```{language}\n(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

    if matches:
        return matches[0].strip()

    code_block_pattern = r"```(?:\w+\n)?(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

    if matches:
        return matches[0].strip()

    return None


def format_diff_summary(diff_blocks: List[Tuple[str, str]]) -> str:
    """
    Create a human-readable summary of diff blocks.

    Args:
        diff_blocks: Parsed diff blocks.

    Returns:
        Summary string.
    """
    if not diff_blocks:
        return "No changes"

    summary = []

    for i, (search_text, replace_text) in enumerate(diff_blocks):
        search_lines = search_text.strip().split("\n")
        replace_lines = replace_text.strip().split("\n")

        if len(search_lines) == 1 and len(replace_lines) == 1:
            summary.append(f"Edit {i+1}: '{search_lines[0][:50]}...' -> '{replace_lines[0][:50]}...'")
        else:
            summary.append(
                f"Edit {i+1}: replace {len(search_lines)} lines with {len(replace_lines)} lines"
            )

    return "\n".join(summary)
