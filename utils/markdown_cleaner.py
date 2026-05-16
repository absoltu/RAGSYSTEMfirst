import re


class MarkdownCleaner:

    @staticmethod
    def clean(text: str) -> str:
        """Clean markdown text while preserving structure and formatting."""
        
        # =====================================
        # normalize newlines
        # =====================================
        text = text.replace("\r\n", "\n")

        # =====================================
        # remove page separators (---)
        # =====================================
        text = re.sub(r"^-{3,}$", "", text, flags=re.MULTILINE)

        # =====================================
        # remove standalone page numbers
        # Only remove 1-3 digit numbers on their own line
        # =====================================
        text = re.sub(
            r"^\s*(?:\d{1,3})\s*$",
            "",
            text,
            flags=re.MULTILINE
        )

        # =====================================
        # fix broken hyphenated words
        # =====================================
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

        # =====================================
        # handle formula placeholders
        # With formula enrichment enabled, formulas should be properly recognized
        # Keep formula placeholders for now, but make them more readable
        # =====================================
        text = re.sub(
            r"<!-- formula-not-decoded -->",
            "[Математическая формула]",
            text
        )

        # =====================================
        # remove excessive empty lines
        # Keep max 2 newlines (= 1 blank line)
        # =====================================
        text = re.sub(r"\n{3,}", "\n\n", text)

        # =====================================
        # normalize spaces within lines
        # but preserve leading spaces (indentation)
        # =====================================
        lines = []
        for line in text.split("\n"):
            # Preserve leading whitespace
            leading_ws = len(line) - len(line.lstrip())
            content = line.lstrip()
            
            if not content:
                lines.append("")
                continue
            
            # Normalize internal spaces only
            content = re.sub(r"[ \t]+", " ", content)
            
            # Restore leading whitespace
            lines.append(" " * leading_ws + content)

        text = "\n".join(lines)

        return text.strip()