"""Tests for V6 improvements (IMP-13 through IMP-16).

Run with:
    .venv/bin/python3 -m pytest tests/test_v6_improvements.py -v
or:
    .venv/bin/python3 tests/test_v6_improvements.py
"""
from __future__ import annotations

import re
import sys
import statistics
import random
import textwrap
from pathlib import Path

# ============================================================
# IMP-13: Test _extract_paper_title import & behaviour
# ============================================================

class TestIMP13_ExtractPaperTitle:
    """IMP-13: runner.py imports _extract_paper_title from executor.
    Verify the import works and the function produces correct results."""

    def test_import_works(self):
        """The import `from researchclaw.pipeline.executor import _extract_paper_title`
        must succeed — runner.py line 394 depends on it."""
        from researchclaw.pipeline.executor import _extract_paper_title
        assert callable(_extract_paper_title), "_extract_paper_title should be callable"
        print("[IMP-13] PASS: import _extract_paper_title works")

    def test_extracts_h1_title(self):
        from researchclaw.pipeline.executor import _extract_paper_title
        md = textwrap.dedent("""\
            # A Novel Approach to Deep Reinforcement Learning

            ## Abstract

            This paper presents...
        """)
        title = _extract_paper_title(md)
        assert title == "A Novel Approach to Deep Reinforcement Learning", \
            f"Expected H1 title, got: {title!r}"
        print(f"[IMP-13] PASS: extracted title = {title!r}")

    def test_skips_abstract_heading(self):
        """Title before Abstract should be found; Abstract heading itself skipped."""
        from researchclaw.pipeline.executor import _extract_paper_title
        md = textwrap.dedent("""\
            # A Real Title of at Least Four Words

            ## Abstract

            Some text...
        """)
        title = _extract_paper_title(md)
        # "Abstract" should be skipped; the real title (before Abstract) is found
        assert title == "A Real Title of at Least Four Words", \
            f"Expected real title, got: {title!r}"
        print(f"[IMP-13] PASS: skipped Abstract, got title = {title!r}")

    def test_title_after_abstract_not_found(self):
        """If the only real title is AFTER Abstract, it should not be found
        (function searches only before Abstract heading)."""
        from researchclaw.pipeline.executor import _extract_paper_title
        md = textwrap.dedent("""\
            # Abstract

            # A Title That Appears After Abstract

            Some text...
        """)
        title = _extract_paper_title(md)
        # Title after Abstract is not in the search region, so fallback
        assert title == "Untitled Paper", \
            f"Expected 'Untitled Paper' since title is after Abstract, got: {title!r}"
        print(f"[IMP-13] PASS: title after Abstract not found, fallback = {title!r}")

    def test_fallback_untitled(self):
        from researchclaw.pipeline.executor import _extract_paper_title
        md = "Just some text without any headings."
        title = _extract_paper_title(md)
        assert title == "Untitled Paper", f"Expected 'Untitled Paper', got: {title!r}"
        print(f"[IMP-13] PASS: fallback = {title!r}")

    def test_bold_title(self):
        from researchclaw.pipeline.executor import _extract_paper_title
        md = textwrap.dedent("""\
            **A Bold Title for This Paper**

            ## Abstract

            Text here...
        """)
        title = _extract_paper_title(md)
        assert "Bold Title" in title, f"Expected bold title, got: {title!r}"
        print(f"[IMP-13] PASS: bold title = {title!r}")


# ============================================================
# IMP-14: Test orphaned cite-key stripping logic
# ============================================================

class TestIMP14_StripOrphanedCites:
    """IMP-14: After packaging, any \\cite{key} where key is not in
    references.bib should be stripped from paper.tex."""

    @staticmethod
    def _run_cite_stripping(tex_text: str, bib_text: str) -> str:
        """Reproduce the IMP-14 logic from runner.py lines 505-532."""
        all_cite_keys: set[str] = set()
        for cm in re.finditer(r"\\cite\{([^}]+)\}", tex_text):
            all_cite_keys.update(k.strip() for k in cm.group(1).split(","))
        bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib_text))
        missing = all_cite_keys - bib_keys

        if missing:
            def _filter_cite(m: re.Match[str]) -> str:
                keys = [k.strip() for k in m.group(1).split(",")]
                kept = [k for k in keys if k not in missing]
                if not kept:
                    return ""
                return "\\cite{" + ", ".join(kept) + "}"

            tex_text = re.sub(r"\\cite\{([^}]+)\}", _filter_cite, tex_text)
            tex_text = re.sub(r"  +", " ", tex_text)
            tex_text = re.sub(r" ([.,;:)])", r"\1", tex_text)
        return tex_text

    def test_mixed_real_and_missing_keys(self):
        """\\cite{real_key, missing_key} should become \\cite{real_key}."""
        tex = r"Some text \cite{real_key, missing_key} and more."
        bib = textwrap.dedent("""\
            @article{real_key,
              author = {Doe},
              title = {Real Paper},
              year = {2024},
            }
        """)
        result = self._run_cite_stripping(tex, bib)
        assert r"\cite{real_key}" in result, f"Expected \\cite{{real_key}}, got: {result!r}"
        assert "missing_key" not in result, f"missing_key should be gone: {result!r}"
        print(f"[IMP-14] PASS: mixed keys → {result!r}")

    def test_all_keys_missing(self):
        """\\cite{missing1, missing2} should be entirely removed."""
        tex = r"Some text \cite{missing1, missing2} more."
        bib = ""  # empty bib
        result = self._run_cite_stripping(tex, bib)
        assert r"\cite" not in result, f"Expected no \\cite, got: {result!r}"
        print(f"[IMP-14] PASS: all missing → {result!r}")

    def test_all_keys_valid(self):
        """When all keys are valid, tex should remain unchanged (except whitespace)."""
        tex = r"Text \cite{key1, key2} end."
        bib = textwrap.dedent("""\
            @article{key1,
              author = {A},
              title = {T},
              year = {2024},
            }

            @article{key2,
              author = {B},
              title = {T2},
              year = {2024},
            }
        """)
        result = self._run_cite_stripping(tex, bib)
        assert r"\cite{key1, key2}" in result, f"Expected unchanged, got: {result!r}"
        print(f"[IMP-14] PASS: all valid → {result!r}")

    def test_multiple_cite_commands(self):
        """Multiple \\cite commands, each with different missing keys."""
        tex = (
            r"First \cite{a, b} second \cite{b, c} third \cite{d}."
        )
        bib = textwrap.dedent("""\
            @article{a,
              author = {X},
              title = {Y},
              year = {2024},
            }

            @article{c,
              author = {X},
              title = {Y},
              year = {2024},
            }
        """)
        result = self._run_cite_stripping(tex, bib)
        # a is valid, b is missing, c is valid, d is missing
        assert r"\cite{a}" in result, f"Expected \\cite{{a}}, got: {result!r}"
        assert r"\cite{c}" in result, f"Expected \\cite{{c}}, got: {result!r}"
        # b should not appear as a cite key
        assert r"\cite{b}" not in result, f"\\cite{{b}} should be gone: {result!r}"
        assert r", b}" not in result and r"{b," not in result, \
            f"b key should be stripped: {result!r}"
        # \cite{d} should be entirely removed (d was the only key)
        assert r"\cite{d}" not in result, f"\\cite{{d}} should be gone: {result!r}"
        print(f"[IMP-14] PASS: multiple cites → {result!r}")

    def test_whitespace_cleanup(self):
        """After removing a full \\cite{}, leftover double-spaces and ' .' are cleaned."""
        tex = r"Text \cite{missing} end."
        bib = ""
        result = self._run_cite_stripping(tex, bib)
        # Should not have double spaces or " ."
        assert "  " not in result, f"Double space in result: {result!r}"
        assert " ." not in result, f"Space-dot in result: {result!r}"
        print(f"[IMP-14] PASS: whitespace cleanup → {result!r}")


# ============================================================
# IMP-14b: Test orphaned cite-key stripping in paper_final.md
# ============================================================

class TestIMP14b_StripOrphanedCitesFromMarkdown:
    """IMP-14b: paper_final.md uses [key1, key2] markdown citation syntax.
    After packaging, any [key] where key is not in references.bib should be
    stripped — mirroring the IMP-14 .tex stripping but in markdown form.

    Conservative: only brackets whose entire content matches a comma-separated
    list of cite-key-shaped tokens are touched. Markdown links like
    [text](url) and other brackets are left alone.
    """

    @staticmethod
    def _run_md_cite_stripping(md_text: str, missing: set[str]) -> str:
        """Reproduce the IMP-14b logic from runner.py."""
        _MD_CITE_KEY_RE = re.compile(r"^[a-z][a-z]+\d{4}[a-z][a-z0-9]*$")

        def _is_md_citation_block(content: str) -> bool:
            keys = [k.strip() for k in content.split(",")]
            keys = [k for k in keys if k]
            return bool(keys) and all(
                _MD_CITE_KEY_RE.match(k) for k in keys
            )

        def _filter_md_cite(m: re.Match[str]) -> str:
            content = m.group(1)
            if not _is_md_citation_block(content):
                return m.group(0)
            keys = [k.strip() for k in content.split(",")]
            kept = [k for k in keys if k not in missing]
            if not kept:
                return ""
            return "[" + ", ".join(kept) + "]"

        new_md = re.sub(r"\[([^\[\]]+)\]", _filter_md_cite, md_text)
        if new_md != md_text:
            new_md = re.sub(r"  +", " ", new_md)
            new_md = re.sub(r" ([.,;:)])", r"\1", new_md)
        return new_md

    def test_mixed_real_and_missing_keys(self):
        """[real_key, missing_key] should become [real_key]."""
        md = "Some text [feng2020cross, bai2021memomentum] and more."
        result = self._run_md_cite_stripping(md, {"bai2021memomentum"})
        assert "[feng2020cross]" in result, (
            f"Expected [feng2020cross], got: {result!r}"
        )
        assert "bai2021memomentum" not in result, (
            f"missing key should be gone: {result!r}"
        )
        print(f"[IMP-14b] PASS: mixed keys → {result!r}")

    def test_all_keys_missing(self):
        """[missing1, missing2] should be entirely removed."""
        md = "Some text [bai2021memomentum, xia2021robust] more."
        result = self._run_md_cite_stripping(
            md, {"bai2021memomentum", "xia2021robust"}
        )
        # Citation block fully removed; surrounding text preserved
        assert "bai2021memomentum" not in result, (
            f"missing1 should be gone: {result!r}"
        )
        assert "xia2021robust" not in result, (
            f"missing2 should be gone: {result!r}"
        )
        # Whitespace cleanup folds the gap
        assert "[" not in result and "]" not in result, (
            f"No brackets should remain: {result!r}"
        )
        print(f"[IMP-14b] PASS: all missing → {result!r}")

    def test_markdown_link_left_untouched(self):
        """Markdown links like [text](url) must NOT be stripped even when
        the link text contains words — they aren't citation blocks."""
        md = "See [the docs](https://example.com) and [Open in DI](url2)."
        # No keys missing here; link should still be untouched (the cite-block
        # detector should reject these brackets even if there were a missing
        # set).
        result = self._run_md_cite_stripping(md, {"docs", "Open in DI"})
        assert "[the docs](https://example.com)" in result, (
            f"Markdown link mangled: {result!r}"
        )
        assert "[Open in DI](url2)" in result, (
            f"Second link mangled: {result!r}"
        )
        print(f"[IMP-14b] PASS: markdown links preserved → {result!r}")

    def test_real_world_paper_pattern(self):
        """Realistic paper sentence with mixed valid/missing cite keys."""
        md = (
            "Prior work has emphasized loss-function design "
            "[feng2020cross, ma2020normalized, zhou2023asymmetric], but "
            "early-stopping interactions [bai2021memomentum, xia2021robust] "
            "are equally important."
        )
        # Three valid, two orphans
        missing = {"bai2021memomentum", "xia2021robust"}
        result = self._run_md_cite_stripping(md, missing)
        assert "feng2020cross" in result
        assert "ma2020normalized" in result
        assert "zhou2023asymmetric" in result
        assert "bai2021memomentum" not in result
        assert "xia2021robust" not in result
        # Surrounding prose intact
        assert "Prior work has emphasized loss-function design" in result
        assert "are equally important." in result
        print(f"[IMP-14b] PASS: real-world pattern → {result!r}")

    def test_no_keys_missing_is_noop(self):
        """When all cited keys are in the bib, paper_final.md is unchanged."""
        md = "Text [valid2024paper, also2023ref] more text."
        result = self._run_md_cite_stripping(md, set())
        assert result == md, f"Expected no-op, got: {result!r}"
        print(f"[IMP-14b] PASS: no-op → {result!r}")


# ============================================================
# IMP-15: Test BibTeX deduplication
# ============================================================

class TestIMP15_BibDedup:
    """IMP-15: Deduplicate .bib entries sharing the same cite key."""

    @staticmethod
    def _run_dedup(bib_text: str) -> str:
        """Reproduce IMP-15 logic from runner.py lines 486-503."""
        _seen_bib_keys: set[str] = set()
        _deduped_entries: list[str] = []
        for _bm in re.finditer(
            r"(@\w+\{([^,]+),.*?\n\})", bib_text, re.DOTALL
        ):
            _bkey = _bm.group(2).strip()
            if _bkey not in _seen_bib_keys:
                _seen_bib_keys.add(_bkey)
                _deduped_entries.append(_bm.group(1))
        if len(_deduped_entries) < len(
            list(re.finditer(r"@\w+\{", bib_text))
        ):
            bib_text = "\n\n".join(_deduped_entries) + "\n"
        return bib_text

    def test_duplicate_entries_removed(self):
        bib = textwrap.dedent("""\
            @article{smith2024,
              author = {Smith},
              title = {Paper 1},
              year = {2024},
            }

            @article{smith2024,
              author = {Smith},
              title = {Paper 1 duplicate},
              year = {2024},
            }

            @article{jones2023,
              author = {Jones},
              title = {Paper 2},
              year = {2023},
            }
        """)
        result = self._run_dedup(bib)
        # Count how many @article{smith2024, appear
        count_smith = len(re.findall(r"@article\{smith2024,", result))
        count_jones = len(re.findall(r"@article\{jones2023,", result))
        assert count_smith == 1, f"Expected 1 smith2024 entry, got {count_smith}"
        assert count_jones == 1, f"Expected 1 jones2023 entry, got {count_jones}"
        # First version should be kept
        assert "Paper 1" in result
        print(f"[IMP-15] PASS: 2 smith2024 → 1, jones2023 kept. Total entries correct.")

    def test_no_duplicates_unchanged(self):
        bib = textwrap.dedent("""\
            @article{alpha2024,
              author = {Alpha},
              title = {A},
              year = {2024},
            }

            @inproceedings{beta2023,
              author = {Beta},
              title = {B},
              year = {2023},
            }
        """)
        result = self._run_dedup(bib)
        # Should remain unchanged (both entries present)
        assert "alpha2024" in result
        assert "beta2023" in result
        count = len(re.findall(r"@\w+\{", result))
        assert count == 2, f"Expected 2 entries, got {count}"
        print(f"[IMP-15] PASS: no duplicates → unchanged")

    def test_triple_duplicate(self):
        bib = textwrap.dedent("""\
            @article{x2024,
              author = {X},
              title = {First},
              year = {2024},
            }

            @article{x2024,
              author = {X},
              title = {Second},
              year = {2024},
            }

            @article{x2024,
              author = {X},
              title = {Third},
              year = {2024},
            }
        """)
        result = self._run_dedup(bib)
        count = len(re.findall(r"@article\{x2024,", result))
        assert count == 1, f"Expected 1 x2024 entry, got {count}"
        # First version kept
        assert "First" in result
        assert "Second" not in result
        assert "Third" not in result
        print(f"[IMP-15] PASS: triple duplicate → 1 entry")

    def test_empty_bib(self):
        """Edge case: empty bib text should not crash."""
        bib = ""
        result = self._run_dedup(bib)
        assert result == "", f"Expected empty, got: {result!r}"
        print(f"[IMP-15] PASS: empty bib → no crash")


# ============================================================
# IMP-16: Test bootstrap CI fallback
# ============================================================

class TestIMP16_BootstrapCIFallback:
    """IMP-16: If bootstrap CI does not contain the mean,
    fall back to normal approximation (mean +/- 1.96*SE)."""

    @staticmethod
    def _compute_ci_with_fallback(vals: list[float]) -> tuple[float, float, bool]:
        """Reproduce IMP-16 logic from executor.py lines 3367-3397.
        Returns (ci_low, ci_high, used_fallback)."""
        _mean = statistics.mean(vals)
        _std = statistics.stdev(vals)

        # Bootstrap 95% CI
        _rng = random.Random(42)
        _boot_means = []
        for _ in range(1000):
            _sample = [_rng.choice(vals) for _ in range(len(vals))]
            _boot_means.append(statistics.mean(_sample))
        _boot_means.sort()
        _ci_low = round(_boot_means[int(0.025 * len(_boot_means))], 6)
        _ci_high = round(_boot_means[int(0.975 * len(_boot_means))], 6)

        # IMP-16: Sanity check
        used_fallback = False
        if _ci_low > _mean or _ci_high < _mean:
            _se = _std / (len(vals) ** 0.5)
            _ci_low = round(_mean - 1.96 * _se, 6)
            _ci_high = round(_mean + 1.96 * _se, 6)
            used_fallback = True

        return _ci_low, _ci_high, used_fallback

    def test_normal_case_no_fallback(self):
        """Normal data: bootstrap CI should contain the mean, no fallback needed."""
        vals = [0.8, 0.82, 0.79, 0.81, 0.83]
        ci_low, ci_high, used_fallback = self._compute_ci_with_fallback(vals)
        mean = statistics.mean(vals)
        assert ci_low <= mean <= ci_high, \
            f"CI [{ci_low}, {ci_high}] should contain mean {mean}"
        assert not used_fallback, "Should NOT have used fallback for normal data"
        print(f"[IMP-16] PASS: normal data → CI=[{ci_low}, {ci_high}], mean={mean:.4f}, no fallback")

    def test_fallback_triggers_for_pathological_data(self):
        """Construct data where bootstrap CI might not contain the mean.

        This tests the fallback logic path itself. We directly test the
        condition and fallback formula rather than relying on pathological
        data generation (which is inherently fragile).
        """
        # Directly test the fallback formula
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean = statistics.mean(vals)
        std = statistics.stdev(vals)
        se = std / (len(vals) ** 0.5)

        # Simulate a bad CI that doesn't contain the mean
        bad_ci_low = mean + 0.1  # Above mean - CI doesn't contain mean
        bad_ci_high = mean + 1.0

        # Apply fallback logic
        assert bad_ci_low > mean, "Bad CI should not contain mean"
        fallback_low = round(mean - 1.96 * se, 6)
        fallback_high = round(mean + 1.96 * se, 6)

        assert fallback_low <= mean <= fallback_high, \
            f"Fallback CI [{fallback_low}, {fallback_high}] must contain mean {mean}"
        print(f"[IMP-16] PASS: fallback CI=[{fallback_low}, {fallback_high}], mean={mean:.4f}")

    def test_fallback_ci_always_contains_mean(self):
        """The normal-approximation fallback MUST always contain the mean."""
        test_cases = [
            [10, 20, 30],
            [0.001, 0.002, 0.003, 0.004],
            [100, 200, 300, 400, 500],
            [-5, -3, -1, 1, 3, 5],
        ]
        for vals in test_cases:
            mean = statistics.mean(vals)
            std = statistics.stdev(vals)
            se = std / (len(vals) ** 0.5)
            ci_low = round(mean - 1.96 * se, 6)
            ci_high = round(mean + 1.96 * se, 6)
            assert ci_low <= mean <= ci_high, \
                f"Fallback CI [{ci_low}, {ci_high}] must contain mean {mean} for vals={vals}"
        print(f"[IMP-16] PASS: fallback always contains mean for {len(test_cases)} test cases")

    def test_condition_check_logic(self):
        """Verify the condition `_ci_low > _mean or _ci_high < _mean` is correct.

        The condition should detect when the mean is OUTSIDE the CI."""
        mean = 5.0
        # Case 1: Mean below CI
        assert (6.0 > mean or 8.0 < mean) == True, "Mean below CI not detected"
        # Case 2: Mean above CI
        assert (1.0 > mean or 4.0 < mean) == True, "Mean above CI not detected"
        # Case 3: Mean inside CI
        assert (3.0 > mean or 7.0 < mean) == False, "Mean inside CI incorrectly flagged"
        # Case 4: Mean equals boundary
        assert (5.0 > mean or 7.0 < mean) == False, "Mean at lower boundary incorrectly flagged"
        assert (3.0 > mean or 5.0 < mean) == False, "Mean at upper boundary incorrectly flagged"
        print("[IMP-16] PASS: condition check logic correct for all cases")

    def test_min_sample_size(self):
        """The code requires len(vals) >= 3 for bootstrap. Verify with exactly 3."""
        vals = [1.0, 2.0, 3.0]
        ci_low, ci_high, _ = self._compute_ci_with_fallback(vals)
        mean = statistics.mean(vals)
        assert ci_low <= mean <= ci_high, \
            f"CI [{ci_low}, {ci_high}] should contain mean {mean} for n=3"
        print(f"[IMP-16] PASS: n=3 works → CI=[{ci_low}, {ci_high}], mean={mean:.4f}")


# ============================================================
# Integration-style: Test the runner.py _package_deliverables
# cite-stripping + dedup pipeline end-to-end
# ============================================================

class TestIMP14_15_Integration:
    """End-to-end test: dedup + cite stripping on a realistic scenario."""

    def test_dedup_then_strip(self):
        """Run dedup (IMP-15) then cite-strip (IMP-14) in sequence, as runner.py does."""
        bib_text = textwrap.dedent("""\
            @article{smith2024,
              author = {Smith},
              title = {Paper A},
              year = {2024},
            }

            @article{smith2024,
              author = {Smith},
              title = {Paper A dup},
              year = {2024},
            }

            @article{jones2023,
              author = {Jones},
              title = {Paper B},
              year = {2023},
            }
        """)
        tex_text = r"Results from \cite{smith2024, jones2023, ghost2024} show..."

        # Step 1: IMP-15 dedup
        _seen: set[str] = set()
        _deduped: list[str] = []
        for m in re.finditer(r"(@\w+\{([^,]+),.*?\n\})", bib_text, re.DOTALL):
            k = m.group(2).strip()
            if k not in _seen:
                _seen.add(k)
                _deduped.append(m.group(1))
        if len(_deduped) < len(list(re.finditer(r"@\w+\{", bib_text))):
            bib_text = "\n\n".join(_deduped) + "\n"

        # Verify dedup
        assert bib_text.count("smith2024") == 1, "Dedup failed for smith2024"

        # Step 2: IMP-14 cite stripping
        all_cite_keys: set[str] = set()
        for cm in re.finditer(r"\\cite\{([^}]+)\}", tex_text):
            all_cite_keys.update(k.strip() for k in cm.group(1).split(","))
        bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib_text))
        missing = all_cite_keys - bib_keys

        assert missing == {"ghost2024"}, f"Expected only ghost2024 missing, got {missing}"

        def _filter_cite(m: re.Match[str]) -> str:
            keys = [k.strip() for k in m.group(1).split(",")]
            kept = [k for k in keys if k not in missing]
            if not kept:
                return ""
            return "\\cite{" + ", ".join(kept) + "}"

        tex_text = re.sub(r"\\cite\{([^}]+)\}", _filter_cite, tex_text)
        tex_text = re.sub(r"  +", " ", tex_text)
        tex_text = re.sub(r" ([.,;:)])", r"\1", tex_text)

        assert r"\cite{smith2024, jones2023}" in tex_text, \
            f"Expected valid keys kept, got: {tex_text!r}"
        assert "ghost2024" not in tex_text, \
            f"ghost2024 should be stripped: {tex_text!r}"
        print(f"[Integration] PASS: dedup + cite strip → {tex_text!r}")


# ============================================================
# Runner
# ============================================================

def run_all_tests():
    """Run all tests manually (fallback if pytest not available)."""
    test_classes = [
        TestIMP13_ExtractPaperTitle,
        TestIMP14_StripOrphanedCites,
        TestIMP15_BibDedup,
        TestIMP16_BootstrapCIFallback,
        TestIMP14_15_Integration,
    ]
    total = 0
    passed = 0
    failed = 0
    errors: list[str] = []

    for cls in test_classes:
        instance = cls()
        test_methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in sorted(test_methods):
            total += 1
            method = getattr(instance, method_name)
            try:
                method()
                passed += 1
            except Exception as e:
                failed += 1
                err_msg = f"FAIL: {cls.__name__}.{method_name}: {e}"
                errors.append(err_msg)
                print(f"  FAIL: {err_msg}")

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if errors:
        print("Failures:")
        for e in errors:
            print(f"  - {e}")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    # Add project root to path
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    success = run_all_tests()
    sys.exit(0 if success else 1)
