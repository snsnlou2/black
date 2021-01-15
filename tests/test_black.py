
import multiprocessing
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
import inspect
from io import BytesIO, TextIOWrapper
import os
from pathlib import Path
from platform import system
import regex as re
import sys
from tempfile import TemporaryDirectory
import types
from typing import Any, BinaryIO, Callable, Dict, Generator, List, Iterator, TypeVar
import unittest
from unittest.mock import patch, MagicMock
import click
from click import unstyle
from click.testing import CliRunner
import black
from black import Feature, TargetVersion
from pathspec import PathSpec
from tests.util import THIS_DIR, read_data, DETERMINISTIC_HEADER, BlackBaseTestCase, DEFAULT_MODE, fs, ff, dump_to_stderr
from .test_primer import PrimerCLITests
THIS_FILE = Path(__file__)
PY36_VERSIONS = {TargetVersion.PY36, TargetVersion.PY37, TargetVersion.PY38, TargetVersion.PY39}
PY36_ARGS = [f'--target-version={version.name.lower()}' for version in PY36_VERSIONS]
T = TypeVar('T')
R = TypeVar('R')

@contextmanager
def cache_dir(exists=True):
    with TemporaryDirectory() as workspace:
        cache_dir = Path(workspace)
        if (not exists):
            cache_dir = (cache_dir / 'new')
        with patch('black.CACHE_DIR', cache_dir):
            (yield cache_dir)

@contextmanager
def event_loop():
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        (yield)
    finally:
        loop.close()

class FakeContext(click.Context):
    'A fake click Context for when calling functions that need it.'

    def __init__(self):
        self.default_map: Dict[(str, Any)] = {}

class FakeParameter(click.Parameter):
    'A fake click Parameter for when calling functions that need it.'

    def __init__(self):
        pass

class BlackRunner(CliRunner):
    'Modify CliRunner so that stderr is not merged with stdout.\n\n    This is a hack that can be removed once we depend on Click 7.x'

    def __init__(self):
        self.stderrbuf = BytesIO()
        self.stdoutbuf = BytesIO()
        self.stdout_bytes = b''
        self.stderr_bytes = b''
        super().__init__()

    @contextmanager
    def isolation(self, *args, **kwargs):
        with super().isolation(*args, **kwargs) as output:
            try:
                hold_stderr = sys.stderr
                sys.stderr = TextIOWrapper(self.stderrbuf, encoding=self.charset)
                (yield output)
            finally:
                self.stdout_bytes = sys.stdout.buffer.getvalue()
                self.stderr_bytes = sys.stderr.buffer.getvalue()
                sys.stderr = hold_stderr

class BlackTestCase(BlackBaseTestCase):

    def invokeBlack(self, args, exit_code=0, ignore_config=True):
        runner = BlackRunner()
        if ignore_config:
            args = ['--verbose', '--config', str((THIS_DIR / 'empty.toml')), *args]
        result = runner.invoke(black.main, args)
        self.assertEqual(result.exit_code, exit_code, msg=f'''Failed with args: {args}
stdout: {runner.stdout_bytes.decode()!r}
stderr: {runner.stderr_bytes.decode()!r}
exception: {result.exception}''')

    @patch('black.dump_to_file', dump_to_stderr)
    def test_empty(self):
        source = expected = ''
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        black.assert_equivalent(source, actual)
        black.assert_stable(source, actual, DEFAULT_MODE)

    def test_empty_ff(self):
        expected = ''
        tmp_file = Path(black.dump_to_file())
        try:
            self.assertFalse(ff(tmp_file, write_back=black.WriteBack.YES))
            with open(tmp_file, encoding='utf8') as f:
                actual = f.read()
        finally:
            os.unlink(tmp_file)
        self.assertFormatEqual(expected, actual)

    def test_piping(self):
        (source, expected) = read_data('src/black/__init__', data=False)
        result = BlackRunner().invoke(black.main, ['-', '--fast', f'--line-length={black.DEFAULT_LINE_LENGTH}'], input=BytesIO(source.encode('utf8')))
        self.assertEqual(result.exit_code, 0)
        self.assertFormatEqual(expected, result.output)
        black.assert_equivalent(source, result.output)
        black.assert_stable(source, result.output, DEFAULT_MODE)

    def test_piping_diff(self):
        diff_header = re.compile('(STDIN|STDOUT)\\t\\d\\d\\d\\d-\\d\\d-\\d\\d \\d\\d:\\d\\d:\\d\\d\\.\\d\\d\\d\\d\\d\\d \\+\\d\\d\\d\\d')
        (source, _) = read_data('expression.py')
        (expected, _) = read_data('expression.diff')
        config = ((THIS_DIR / 'data') / 'empty_pyproject.toml')
        args = ['-', '--fast', f'--line-length={black.DEFAULT_LINE_LENGTH}', '--diff', f'--config={config}']
        result = BlackRunner().invoke(black.main, args, input=BytesIO(source.encode('utf8')))
        self.assertEqual(result.exit_code, 0)
        actual = diff_header.sub(DETERMINISTIC_HEADER, result.output)
        actual = (actual.rstrip() + '\n')
        self.assertEqual(expected, actual)

    def test_piping_diff_with_color(self):
        (source, _) = read_data('expression.py')
        config = ((THIS_DIR / 'data') / 'empty_pyproject.toml')
        args = ['-', '--fast', f'--line-length={black.DEFAULT_LINE_LENGTH}', '--diff', '--color', f'--config={config}']
        result = BlackRunner().invoke(black.main, args, input=BytesIO(source.encode('utf8')))
        actual = result.output
        self.assertIn('\x1b[1;37m', actual)
        self.assertIn('\x1b[36m', actual)
        self.assertIn('\x1b[32m', actual)
        self.assertIn('\x1b[31m', actual)
        self.assertIn('\x1b[0m', actual)

    @patch('black.dump_to_file', dump_to_stderr)
    def _test_wip(self):
        (source, expected) = read_data('wip')
        sys.settrace(tracefunc)
        mode = replace(DEFAULT_MODE, experimental_string_processing=False, target_versions={black.TargetVersion.PY38})
        actual = fs(source, mode=mode)
        sys.settrace(None)
        self.assertFormatEqual(expected, actual)
        black.assert_equivalent(source, actual)
        black.assert_stable(source, actual, black.FileMode())

    @unittest.expectedFailure
    @patch('black.dump_to_file', dump_to_stderr)
    def test_trailing_comma_optional_parens_stability1(self):
        (source, _expected) = read_data('trailing_comma_optional_parens1')
        actual = fs(source)
        black.assert_stable(source, actual, DEFAULT_MODE)

    @unittest.expectedFailure
    @patch('black.dump_to_file', dump_to_stderr)
    def test_trailing_comma_optional_parens_stability2(self):
        (source, _expected) = read_data('trailing_comma_optional_parens2')
        actual = fs(source)
        black.assert_stable(source, actual, DEFAULT_MODE)

    @unittest.expectedFailure
    @patch('black.dump_to_file', dump_to_stderr)
    def test_trailing_comma_optional_parens_stability3(self):
        (source, _expected) = read_data('trailing_comma_optional_parens3')
        actual = fs(source)
        black.assert_stable(source, actual, DEFAULT_MODE)

    @patch('black.dump_to_file', dump_to_stderr)
    def test_pep_572(self):
        (source, expected) = read_data('pep_572')
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        black.assert_stable(source, actual, DEFAULT_MODE)
        if (sys.version_info >= (3, 8)):
            black.assert_equivalent(source, actual)

    def test_pep_572_version_detection(self):
        (source, _) = read_data('pep_572')
        root = black.lib2to3_parse(source)
        features = black.get_features_used(root)
        self.assertIn(black.Feature.ASSIGNMENT_EXPRESSIONS, features)
        versions = black.detect_target_versions(root)
        self.assertIn(black.TargetVersion.PY38, versions)

    def test_expression_ff(self):
        (source, expected) = read_data('expression')
        tmp_file = Path(black.dump_to_file(source))
        try:
            self.assertTrue(ff(tmp_file, write_back=black.WriteBack.YES))
            with open(tmp_file, encoding='utf8') as f:
                actual = f.read()
        finally:
            os.unlink(tmp_file)
        self.assertFormatEqual(expected, actual)
        with patch('black.dump_to_file', dump_to_stderr):
            black.assert_equivalent(source, actual)
            black.assert_stable(source, actual, DEFAULT_MODE)

    def test_expression_diff(self):
        (source, _) = read_data('expression.py')
        (expected, _) = read_data('expression.diff')
        tmp_file = Path(black.dump_to_file(source))
        diff_header = re.compile(f'{re.escape(str(tmp_file))}\t\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d\.\d\d\d\d\d\d \+\d\d\d\d')
        try:
            result = BlackRunner().invoke(black.main, ['--diff', str(tmp_file)])
            self.assertEqual(result.exit_code, 0)
        finally:
            os.unlink(tmp_file)
        actual = result.output
        actual = diff_header.sub(DETERMINISTIC_HEADER, actual)
        actual = (actual.rstrip() + '\n')
        if (expected != actual):
            dump = black.dump_to_file(actual)
            msg = f"Expected diff isn't equal to the actual. If you made changes to expression.py and this is an anticipated difference, overwrite tests/data/expression.diff with {dump}"
            self.assertEqual(expected, actual, msg)

    def test_expression_diff_with_color(self):
        (source, _) = read_data('expression.py')
        (expected, _) = read_data('expression.diff')
        tmp_file = Path(black.dump_to_file(source))
        try:
            result = BlackRunner().invoke(black.main, ['--diff', '--color', str(tmp_file)])
        finally:
            os.unlink(tmp_file)
        actual = result.output
        self.assertIn('\x1b[1;37m', actual)
        self.assertIn('\x1b[36m', actual)
        self.assertIn('\x1b[32m', actual)
        self.assertIn('\x1b[31m', actual)
        self.assertIn('\x1b[0m', actual)

    @patch('black.dump_to_file', dump_to_stderr)
    def test_pep_570(self):
        (source, expected) = read_data('pep_570')
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        black.assert_stable(source, actual, DEFAULT_MODE)
        if (sys.version_info >= (3, 8)):
            black.assert_equivalent(source, actual)

    def test_detect_pos_only_arguments(self):
        (source, _) = read_data('pep_570')
        root = black.lib2to3_parse(source)
        features = black.get_features_used(root)
        self.assertIn(black.Feature.POS_ONLY_ARGUMENTS, features)
        versions = black.detect_target_versions(root)
        self.assertIn(black.TargetVersion.PY38, versions)

    @patch('black.dump_to_file', dump_to_stderr)
    def test_string_quotes(self):
        (source, expected) = read_data('string_quotes')
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        black.assert_equivalent(source, actual)
        black.assert_stable(source, actual, DEFAULT_MODE)
        mode = replace(DEFAULT_MODE, string_normalization=False)
        not_normalized = fs(source, mode=mode)
        self.assertFormatEqual(source.replace('\\\n', ''), not_normalized)
        black.assert_equivalent(source, not_normalized)
        black.assert_stable(source, not_normalized, mode=mode)

    @patch('black.dump_to_file', dump_to_stderr)
    def test_docstring_no_string_normalization(self):
        'Like test_docstring but with string normalization off.'
        (source, expected) = read_data('docstring_no_string_normalization')
        mode = replace(DEFAULT_MODE, string_normalization=False)
        actual = fs(source, mode=mode)
        self.assertFormatEqual(expected, actual)
        black.assert_equivalent(source, actual)
        black.assert_stable(source, actual, mode)

    def test_long_strings_flag_disabled(self):
        'Tests for turning off the string processing logic.'
        (source, expected) = read_data('long_strings_flag_disabled')
        mode = replace(DEFAULT_MODE, experimental_string_processing=False)
        actual = fs(source, mode=mode)
        self.assertFormatEqual(expected, actual)
        black.assert_stable(expected, actual, mode)

    @patch('black.dump_to_file', dump_to_stderr)
    def test_numeric_literals(self):
        (source, expected) = read_data('numeric_literals')
        mode = replace(DEFAULT_MODE, target_versions=PY36_VERSIONS)
        actual = fs(source, mode=mode)
        self.assertFormatEqual(expected, actual)
        black.assert_equivalent(source, actual)
        black.assert_stable(source, actual, mode)

    @patch('black.dump_to_file', dump_to_stderr)
    def test_numeric_literals_ignoring_underscores(self):
        (source, expected) = read_data('numeric_literals_skip_underscores')
        mode = replace(DEFAULT_MODE, target_versions=PY36_VERSIONS)
        actual = fs(source, mode=mode)
        self.assertFormatEqual(expected, actual)
        black.assert_equivalent(source, actual)
        black.assert_stable(source, actual, mode)

    @patch('black.dump_to_file', dump_to_stderr)
    def test_python2_print_function(self):
        (source, expected) = read_data('python2_print_function')
        mode = replace(DEFAULT_MODE, target_versions={TargetVersion.PY27})
        actual = fs(source, mode=mode)
        self.assertFormatEqual(expected, actual)
        black.assert_equivalent(source, actual)
        black.assert_stable(source, actual, mode)

    @patch('black.dump_to_file', dump_to_stderr)
    def test_stub(self):
        mode = replace(DEFAULT_MODE, is_pyi=True)
        (source, expected) = read_data('stub.pyi')
        actual = fs(source, mode=mode)
        self.assertFormatEqual(expected, actual)
        black.assert_stable(source, actual, mode)

    @patch('black.dump_to_file', dump_to_stderr)
    def test_async_as_identifier(self):
        source_path = ((THIS_DIR / 'data') / 'async_as_identifier.py').resolve()
        (source, expected) = read_data('async_as_identifier')
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        (major, minor) = sys.version_info[:2]
        if ((major < 3) or ((major <= 3) and (minor < 7))):
            black.assert_equivalent(source, actual)
        black.assert_stable(source, actual, DEFAULT_MODE)
        self.invokeBlack([str(source_path), '--target-version', 'py36'])
        self.invokeBlack([str(source_path), '--target-version', 'py37'], exit_code=123)

    @patch('black.dump_to_file', dump_to_stderr)
    def test_python37(self):
        source_path = ((THIS_DIR / 'data') / 'python37.py').resolve()
        (source, expected) = read_data('python37')
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        (major, minor) = sys.version_info[:2]
        if ((major > 3) or ((major == 3) and (minor >= 7))):
            black.assert_equivalent(source, actual)
        black.assert_stable(source, actual, DEFAULT_MODE)
        self.invokeBlack([str(source_path), '--target-version', 'py37'])
        self.invokeBlack([str(source_path), '--target-version', 'py36'], exit_code=123)

    @patch('black.dump_to_file', dump_to_stderr)
    def test_python38(self):
        (source, expected) = read_data('python38')
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        (major, minor) = sys.version_info[:2]
        if ((major > 3) or ((major == 3) and (minor >= 8))):
            black.assert_equivalent(source, actual)
        black.assert_stable(source, actual, DEFAULT_MODE)

    @patch('black.dump_to_file', dump_to_stderr)
    def test_python39(self):
        (source, expected) = read_data('python39')
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        (major, minor) = sys.version_info[:2]
        if ((major > 3) or ((major == 3) and (minor >= 9))):
            black.assert_equivalent(source, actual)
        black.assert_stable(source, actual, DEFAULT_MODE)

    def test_tab_comment_indentation(self):
        contents_tab = 'if 1:\n\tif 2:\n\t\tpass\n\t# comment\n\tpass\n'
        contents_spc = 'if 1:\n    if 2:\n        pass\n    # comment\n    pass\n'
        self.assertFormatEqual(contents_spc, fs(contents_spc))
        self.assertFormatEqual(contents_spc, fs(contents_tab))
        contents_tab = 'if 1:\n\tif 2:\n\t\tpass\n\t\t# comment\n\tpass\n'
        contents_spc = 'if 1:\n    if 2:\n        pass\n        # comment\n    pass\n'
        self.assertFormatEqual(contents_spc, fs(contents_spc))
        self.assertFormatEqual(contents_spc, fs(contents_tab))
        contents_tab = 'if 1:\n        if 2:\n\t\tpass\n\t# comment\n        pass\n'
        contents_spc = 'if 1:\n    if 2:\n        pass\n    # comment\n    pass\n'
        self.assertFormatEqual(contents_spc, fs(contents_spc))
        self.assertFormatEqual(contents_spc, fs(contents_tab))
        contents_tab = 'if 1:\n        if 2:\n\t\tpass\n\t\t# comment\n        pass\n'
        contents_spc = 'if 1:\n    if 2:\n        pass\n        # comment\n    pass\n'
        self.assertFormatEqual(contents_spc, fs(contents_spc))
        self.assertFormatEqual(contents_spc, fs(contents_tab))

    def test_report_verbose(self):
        report = black.Report(verbose=True)
        out_lines = []
        err_lines = []

        def out(msg: str, **kwargs: Any) -> None:
            out_lines.append(msg)

        def err(msg: str, **kwargs: Any) -> None:
            err_lines.append(msg)
        with patch('black.out', out), patch('black.err', err):
            report.done(Path('f1'), black.Changed.NO)
            self.assertEqual(len(out_lines), 1)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(out_lines[(- 1)], 'f1 already well formatted, good job.')
            self.assertEqual(unstyle(str(report)), '1 file left unchanged.')
            self.assertEqual(report.return_code, 0)
            report.done(Path('f2'), black.Changed.YES)
            self.assertEqual(len(out_lines), 2)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(out_lines[(- 1)], 'reformatted f2')
            self.assertEqual(unstyle(str(report)), '1 file reformatted, 1 file left unchanged.')
            report.done(Path('f3'), black.Changed.CACHED)
            self.assertEqual(len(out_lines), 3)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(out_lines[(- 1)], "f3 wasn't modified on disk since last run.")
            self.assertEqual(unstyle(str(report)), '1 file reformatted, 2 files left unchanged.')
            self.assertEqual(report.return_code, 0)
            report.check = True
            self.assertEqual(report.return_code, 1)
            report.check = False
            report.failed(Path('e1'), 'boom')
            self.assertEqual(len(out_lines), 3)
            self.assertEqual(len(err_lines), 1)
            self.assertEqual(err_lines[(- 1)], 'error: cannot format e1: boom')
            self.assertEqual(unstyle(str(report)), '1 file reformatted, 2 files left unchanged, 1 file failed to reformat.')
            self.assertEqual(report.return_code, 123)
            report.done(Path('f3'), black.Changed.YES)
            self.assertEqual(len(out_lines), 4)
            self.assertEqual(len(err_lines), 1)
            self.assertEqual(out_lines[(- 1)], 'reformatted f3')
            self.assertEqual(unstyle(str(report)), '2 files reformatted, 2 files left unchanged, 1 file failed to reformat.')
            self.assertEqual(report.return_code, 123)
            report.failed(Path('e2'), 'boom')
            self.assertEqual(len(out_lines), 4)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(err_lines[(- 1)], 'error: cannot format e2: boom')
            self.assertEqual(unstyle(str(report)), '2 files reformatted, 2 files left unchanged, 2 files failed to reformat.')
            self.assertEqual(report.return_code, 123)
            report.path_ignored(Path('wat'), 'no match')
            self.assertEqual(len(out_lines), 5)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(out_lines[(- 1)], 'wat ignored: no match')
            self.assertEqual(unstyle(str(report)), '2 files reformatted, 2 files left unchanged, 2 files failed to reformat.')
            self.assertEqual(report.return_code, 123)
            report.done(Path('f4'), black.Changed.NO)
            self.assertEqual(len(out_lines), 6)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(out_lines[(- 1)], 'f4 already well formatted, good job.')
            self.assertEqual(unstyle(str(report)), '2 files reformatted, 3 files left unchanged, 2 files failed to reformat.')
            self.assertEqual(report.return_code, 123)
            report.check = True
            self.assertEqual(unstyle(str(report)), '2 files would be reformatted, 3 files would be left unchanged, 2 files would fail to reformat.')
            report.check = False
            report.diff = True
            self.assertEqual(unstyle(str(report)), '2 files would be reformatted, 3 files would be left unchanged, 2 files would fail to reformat.')

    def test_report_quiet(self):
        report = black.Report(quiet=True)
        out_lines = []
        err_lines = []

        def out(msg: str, **kwargs: Any) -> None:
            out_lines.append(msg)

        def err(msg: str, **kwargs: Any) -> None:
            err_lines.append(msg)
        with patch('black.out', out), patch('black.err', err):
            report.done(Path('f1'), black.Changed.NO)
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(unstyle(str(report)), '1 file left unchanged.')
            self.assertEqual(report.return_code, 0)
            report.done(Path('f2'), black.Changed.YES)
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(unstyle(str(report)), '1 file reformatted, 1 file left unchanged.')
            report.done(Path('f3'), black.Changed.CACHED)
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(unstyle(str(report)), '1 file reformatted, 2 files left unchanged.')
            self.assertEqual(report.return_code, 0)
            report.check = True
            self.assertEqual(report.return_code, 1)
            report.check = False
            report.failed(Path('e1'), 'boom')
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 1)
            self.assertEqual(err_lines[(- 1)], 'error: cannot format e1: boom')
            self.assertEqual(unstyle(str(report)), '1 file reformatted, 2 files left unchanged, 1 file failed to reformat.')
            self.assertEqual(report.return_code, 123)
            report.done(Path('f3'), black.Changed.YES)
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 1)
            self.assertEqual(unstyle(str(report)), '2 files reformatted, 2 files left unchanged, 1 file failed to reformat.')
            self.assertEqual(report.return_code, 123)
            report.failed(Path('e2'), 'boom')
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(err_lines[(- 1)], 'error: cannot format e2: boom')
            self.assertEqual(unstyle(str(report)), '2 files reformatted, 2 files left unchanged, 2 files failed to reformat.')
            self.assertEqual(report.return_code, 123)
            report.path_ignored(Path('wat'), 'no match')
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(unstyle(str(report)), '2 files reformatted, 2 files left unchanged, 2 files failed to reformat.')
            self.assertEqual(report.return_code, 123)
            report.done(Path('f4'), black.Changed.NO)
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(unstyle(str(report)), '2 files reformatted, 3 files left unchanged, 2 files failed to reformat.')
            self.assertEqual(report.return_code, 123)
            report.check = True
            self.assertEqual(unstyle(str(report)), '2 files would be reformatted, 3 files would be left unchanged, 2 files would fail to reformat.')
            report.check = False
            report.diff = True
            self.assertEqual(unstyle(str(report)), '2 files would be reformatted, 3 files would be left unchanged, 2 files would fail to reformat.')

    def test_report_normal(self):
        report = black.Report()
        out_lines = []
        err_lines = []

        def out(msg: str, **kwargs: Any) -> None:
            out_lines.append(msg)

        def err(msg: str, **kwargs: Any) -> None:
            err_lines.append(msg)
        with patch('black.out', out), patch('black.err', err):
            report.done(Path('f1'), black.Changed.NO)
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(unstyle(str(report)), '1 file left unchanged.')
            self.assertEqual(report.return_code, 0)
            report.done(Path('f2'), black.Changed.YES)
            self.assertEqual(len(out_lines), 1)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(out_lines[(- 1)], 'reformatted f2')
            self.assertEqual(unstyle(str(report)), '1 file reformatted, 1 file left unchanged.')
            report.done(Path('f3'), black.Changed.CACHED)
            self.assertEqual(len(out_lines), 1)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(out_lines[(- 1)], 'reformatted f2')
            self.assertEqual(unstyle(str(report)), '1 file reformatted, 2 files left unchanged.')
            self.assertEqual(report.return_code, 0)
            report.check = True
            self.assertEqual(report.return_code, 1)
            report.check = False
            report.failed(Path('e1'), 'boom')
            self.assertEqual(len(out_lines), 1)
            self.assertEqual(len(err_lines), 1)
            self.assertEqual(err_lines[(- 1)], 'error: cannot format e1: boom')
            self.assertEqual(unstyle(str(report)), '1 file reformatted, 2 files left unchanged, 1 file failed to reformat.')
            self.assertEqual(report.return_code, 123)
            report.done(Path('f3'), black.Changed.YES)
            self.assertEqual(len(out_lines), 2)
            self.assertEqual(len(err_lines), 1)
            self.assertEqual(out_lines[(- 1)], 'reformatted f3')
            self.assertEqual(unstyle(str(report)), '2 files reformatted, 2 files left unchanged, 1 file failed to reformat.')
            self.assertEqual(report.return_code, 123)
            report.failed(Path('e2'), 'boom')
            self.assertEqual(len(out_lines), 2)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(err_lines[(- 1)], 'error: cannot format e2: boom')
            self.assertEqual(unstyle(str(report)), '2 files reformatted, 2 files left unchanged, 2 files failed to reformat.')
            self.assertEqual(report.return_code, 123)
            report.path_ignored(Path('wat'), 'no match')
            self.assertEqual(len(out_lines), 2)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(unstyle(str(report)), '2 files reformatted, 2 files left unchanged, 2 files failed to reformat.')
            self.assertEqual(report.return_code, 123)
            report.done(Path('f4'), black.Changed.NO)
            self.assertEqual(len(out_lines), 2)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(unstyle(str(report)), '2 files reformatted, 3 files left unchanged, 2 files failed to reformat.')
            self.assertEqual(report.return_code, 123)
            report.check = True
            self.assertEqual(unstyle(str(report)), '2 files would be reformatted, 3 files would be left unchanged, 2 files would fail to reformat.')
            report.check = False
            report.diff = True
            self.assertEqual(unstyle(str(report)), '2 files would be reformatted, 3 files would be left unchanged, 2 files would fail to reformat.')

    def test_lib2to3_parse(self):
        with self.assertRaises(black.InvalidInput):
            black.lib2to3_parse('invalid syntax')
        straddling = 'x + y'
        black.lib2to3_parse(straddling)
        black.lib2to3_parse(straddling, {TargetVersion.PY27})
        black.lib2to3_parse(straddling, {TargetVersion.PY36})
        black.lib2to3_parse(straddling, {TargetVersion.PY27, TargetVersion.PY36})
        py2_only = 'print x'
        black.lib2to3_parse(py2_only)
        black.lib2to3_parse(py2_only, {TargetVersion.PY27})
        with self.assertRaises(black.InvalidInput):
            black.lib2to3_parse(py2_only, {TargetVersion.PY36})
        with self.assertRaises(black.InvalidInput):
            black.lib2to3_parse(py2_only, {TargetVersion.PY27, TargetVersion.PY36})
        py3_only = 'exec(x, end=y)'
        black.lib2to3_parse(py3_only)
        with self.assertRaises(black.InvalidInput):
            black.lib2to3_parse(py3_only, {TargetVersion.PY27})
        black.lib2to3_parse(py3_only, {TargetVersion.PY36})
        black.lib2to3_parse(py3_only, {TargetVersion.PY27, TargetVersion.PY36})

    def test_get_features_used_decorator(self):
        (simples, relaxed) = read_data('decorators')
        for simple_test in simples.split('##')[1:]:
            node = black.lib2to3_parse(simple_test)
            decorator = str(node.children[0].children[0]).strip()
            self.assertNotIn(Feature.RELAXED_DECORATORS, black.get_features_used(node), msg=f"decorator '{decorator}' follows python<=3.8 syntaxbut is detected as 3.9+")
        for relaxed_test in relaxed.split('##')[1:]:
            node = black.lib2to3_parse(relaxed_test)
            decorator = str(node.children[0].children[0]).strip()
            self.assertIn(Feature.RELAXED_DECORATORS, black.get_features_used(node), msg=f"decorator '{decorator}' uses python3.9+ syntaxbut is detected as python<=3.8")

    def test_get_features_used(self):
        node = black.lib2to3_parse('def f(*, arg): ...\n')
        self.assertEqual(black.get_features_used(node), set())
        node = black.lib2to3_parse('def f(*, arg,): ...\n')
        self.assertEqual(black.get_features_used(node), {Feature.TRAILING_COMMA_IN_DEF})
        node = black.lib2to3_parse('f(*arg,)\n')
        self.assertEqual(black.get_features_used(node), {Feature.TRAILING_COMMA_IN_CALL})
        node = black.lib2to3_parse("def f(*, arg): f'string'\n")
        self.assertEqual(black.get_features_used(node), {Feature.F_STRINGS})
        node = black.lib2to3_parse('123_456\n')
        self.assertEqual(black.get_features_used(node), {Feature.NUMERIC_UNDERSCORES})
        node = black.lib2to3_parse('123456\n')
        self.assertEqual(black.get_features_used(node), set())
        (source, expected) = read_data('function')
        node = black.lib2to3_parse(source)
        expected_features = {Feature.TRAILING_COMMA_IN_CALL, Feature.TRAILING_COMMA_IN_DEF, Feature.F_STRINGS}
        self.assertEqual(black.get_features_used(node), expected_features)
        node = black.lib2to3_parse(expected)
        self.assertEqual(black.get_features_used(node), expected_features)
        (source, expected) = read_data('expression')
        node = black.lib2to3_parse(source)
        self.assertEqual(black.get_features_used(node), set())
        node = black.lib2to3_parse(expected)
        self.assertEqual(black.get_features_used(node), set())

    def test_get_future_imports(self):
        node = black.lib2to3_parse('\n')
        self.assertEqual(set(), black.get_future_imports(node))
        node = black.lib2to3_parse('from __future__ import black\n')
        self.assertEqual({'black'}, black.get_future_imports(node))
        node = black.lib2to3_parse('from __future__ import multiple, imports\n')
        self.assertEqual({'multiple', 'imports'}, black.get_future_imports(node))
        node = black.lib2to3_parse('from __future__ import (parenthesized, imports)\n')
        self.assertEqual({'parenthesized', 'imports'}, black.get_future_imports(node))
        node = black.lib2to3_parse('from __future__ import multiple\nfrom __future__ import imports\n')
        self.assertEqual({'multiple', 'imports'}, black.get_future_imports(node))
        node = black.lib2to3_parse('# comment\nfrom __future__ import black\n')
        self.assertEqual({'black'}, black.get_future_imports(node))
        node = black.lib2to3_parse('"""docstring"""\nfrom __future__ import black\n')
        self.assertEqual({'black'}, black.get_future_imports(node))
        node = black.lib2to3_parse('some(other, code)\nfrom __future__ import black\n')
        self.assertEqual(set(), black.get_future_imports(node))
        node = black.lib2to3_parse('from some.module import black\n')
        self.assertEqual(set(), black.get_future_imports(node))
        node = black.lib2to3_parse('from __future__ import unicode_literals as _unicode_literals')
        self.assertEqual({'unicode_literals'}, black.get_future_imports(node))
        node = black.lib2to3_parse('from __future__ import unicode_literals as _lol, print')
        self.assertEqual({'unicode_literals', 'print'}, black.get_future_imports(node))

    def test_debug_visitor(self):
        (source, _) = read_data('debug_visitor.py')
        (expected, _) = read_data('debug_visitor.out')
        out_lines = []
        err_lines = []

        def out(msg: str, **kwargs: Any) -> None:
            out_lines.append(msg)

        def err(msg: str, **kwargs: Any) -> None:
            err_lines.append(msg)
        with patch('black.out', out), patch('black.err', err):
            black.DebugVisitor.show(source)
        actual = ('\n'.join(out_lines) + '\n')
        log_name = ''
        if (expected != actual):
            log_name = black.dump_to_file(*out_lines)
        self.assertEqual(expected, actual, f'AST print out is different. Actual version dumped to {log_name}')

    def test_format_file_contents(self):
        empty = ''
        mode = DEFAULT_MODE
        with self.assertRaises(black.NothingChanged):
            black.format_file_contents(empty, mode=mode, fast=False)
        just_nl = '\n'
        with self.assertRaises(black.NothingChanged):
            black.format_file_contents(just_nl, mode=mode, fast=False)
        same = 'j = [1, 2, 3]\n'
        with self.assertRaises(black.NothingChanged):
            black.format_file_contents(same, mode=mode, fast=False)
        different = 'j = [1,2,3]'
        expected = same
        actual = black.format_file_contents(different, mode=mode, fast=False)
        self.assertEqual(expected, actual)
        invalid = 'return if you can'
        with self.assertRaises(black.InvalidInput) as e:
            black.format_file_contents(invalid, mode=mode, fast=False)
        self.assertEqual(str(e.exception), 'Cannot parse: 1:7: return if you can')

    def test_endmarker(self):
        n = black.lib2to3_parse('\n')
        self.assertEqual(n.type, black.syms.file_input)
        self.assertEqual(len(n.children), 1)
        self.assertEqual(n.children[0].type, black.token.ENDMARKER)

    @unittest.skipIf(os.environ.get('SKIP_AST_PRINT'), 'user set SKIP_AST_PRINT')
    def test_assertFormatEqual(self):
        out_lines = []
        err_lines = []

        def out(msg: str, **kwargs: Any) -> None:
            out_lines.append(msg)

        def err(msg: str, **kwargs: Any) -> None:
            err_lines.append(msg)
        with patch('black.out', out), patch('black.err', err):
            with self.assertRaises(AssertionError):
                self.assertFormatEqual('j = [1, 2, 3]', 'j = [1, 2, 3,]')
        out_str = ''.join(out_lines)
        self.assertTrue(('Expected tree:' in out_str))
        self.assertTrue(('Actual tree:' in out_str))
        self.assertEqual(''.join(err_lines), '')

    def test_cache_broken_file(self):
        mode = DEFAULT_MODE
        with cache_dir() as workspace:
            cache_file = black.get_cache_file(mode)
            with cache_file.open('w') as fobj:
                fobj.write('this is not a pickle')
            self.assertEqual(black.read_cache(mode), {})
            src = (workspace / 'test.py').resolve()
            with src.open('w') as fobj:
                fobj.write("print('hello')")
            self.invokeBlack([str(src)])
            cache = black.read_cache(mode)
            self.assertIn(src, cache)

    def test_cache_single_file_already_cached(self):
        mode = DEFAULT_MODE
        with cache_dir() as workspace:
            src = (workspace / 'test.py').resolve()
            with src.open('w') as fobj:
                fobj.write("print('hello')")
            black.write_cache({}, [src], mode)
            self.invokeBlack([str(src)])
            with src.open('r') as fobj:
                self.assertEqual(fobj.read(), "print('hello')")

    @event_loop()
    def test_cache_multiple_files(self):
        mode = DEFAULT_MODE
        with cache_dir() as workspace, patch('black.ProcessPoolExecutor', new=ThreadPoolExecutor):
            one = (workspace / 'one.py').resolve()
            with one.open('w') as fobj:
                fobj.write("print('hello')")
            two = (workspace / 'two.py').resolve()
            with two.open('w') as fobj:
                fobj.write("print('hello')")
            black.write_cache({}, [one], mode)
            self.invokeBlack([str(workspace)])
            with one.open('r') as fobj:
                self.assertEqual(fobj.read(), "print('hello')")
            with two.open('r') as fobj:
                self.assertEqual(fobj.read(), 'print("hello")\n')
            cache = black.read_cache(mode)
            self.assertIn(one, cache)
            self.assertIn(two, cache)

    def test_no_cache_when_writeback_diff(self):
        mode = DEFAULT_MODE
        with cache_dir() as workspace:
            src = (workspace / 'test.py').resolve()
            with src.open('w') as fobj:
                fobj.write("print('hello')")
            with patch('black.read_cache') as read_cache, patch('black.write_cache') as write_cache:
                self.invokeBlack([str(src), '--diff'])
                cache_file = black.get_cache_file(mode)
                self.assertFalse(cache_file.exists())
                write_cache.assert_not_called()
                read_cache.assert_not_called()

    def test_no_cache_when_writeback_color_diff(self):
        mode = DEFAULT_MODE
        with cache_dir() as workspace:
            src = (workspace / 'test.py').resolve()
            with src.open('w') as fobj:
                fobj.write("print('hello')")
            with patch('black.read_cache') as read_cache, patch('black.write_cache') as write_cache:
                self.invokeBlack([str(src), '--diff', '--color'])
                cache_file = black.get_cache_file(mode)
                self.assertFalse(cache_file.exists())
                write_cache.assert_not_called()
                read_cache.assert_not_called()

    @event_loop()
    def test_output_locking_when_writeback_diff(self):
        with cache_dir() as workspace:
            for tag in range(0, 4):
                src = (workspace / f'test{tag}.py').resolve()
                with src.open('w') as fobj:
                    fobj.write("print('hello')")
            with patch('black.Manager', wraps=multiprocessing.Manager) as mgr:
                self.invokeBlack(['--diff', str(workspace)], exit_code=0)
                mgr.assert_called()

    @event_loop()
    def test_output_locking_when_writeback_color_diff(self):
        with cache_dir() as workspace:
            for tag in range(0, 4):
                src = (workspace / f'test{tag}.py').resolve()
                with src.open('w') as fobj:
                    fobj.write("print('hello')")
            with patch('black.Manager', wraps=multiprocessing.Manager) as mgr:
                self.invokeBlack(['--diff', '--color', str(workspace)], exit_code=0)
                mgr.assert_called()

    def test_no_cache_when_stdin(self):
        mode = DEFAULT_MODE
        with cache_dir():
            result = CliRunner().invoke(black.main, ['-'], input=BytesIO(b"print('hello')"))
            self.assertEqual(result.exit_code, 0)
            cache_file = black.get_cache_file(mode)
            self.assertFalse(cache_file.exists())

    def test_read_cache_no_cachefile(self):
        mode = DEFAULT_MODE
        with cache_dir():
            self.assertEqual(black.read_cache(mode), {})

    def test_write_cache_read_cache(self):
        mode = DEFAULT_MODE
        with cache_dir() as workspace:
            src = (workspace / 'test.py').resolve()
            src.touch()
            black.write_cache({}, [src], mode)
            cache = black.read_cache(mode)
            self.assertIn(src, cache)
            self.assertEqual(cache[src], black.get_cache_info(src))

    def test_filter_cached(self):
        with TemporaryDirectory() as workspace:
            path = Path(workspace)
            uncached = (path / 'uncached').resolve()
            cached = (path / 'cached').resolve()
            cached_but_changed = (path / 'changed').resolve()
            uncached.touch()
            cached.touch()
            cached_but_changed.touch()
            cache = {cached: black.get_cache_info(cached), cached_but_changed: (0.0, 0)}
            (todo, done) = black.filter_cached(cache, {uncached, cached, cached_but_changed})
            self.assertEqual(todo, {uncached, cached_but_changed})
            self.assertEqual(done, {cached})

    def test_write_cache_creates_directory_if_needed(self):
        mode = DEFAULT_MODE
        with cache_dir(exists=False) as workspace:
            self.assertFalse(workspace.exists())
            black.write_cache({}, [], mode)
            self.assertTrue(workspace.exists())

    @event_loop()
    def test_failed_formatting_does_not_get_cached(self):
        mode = DEFAULT_MODE
        with cache_dir() as workspace, patch('black.ProcessPoolExecutor', new=ThreadPoolExecutor):
            failing = (workspace / 'failing.py').resolve()
            with failing.open('w') as fobj:
                fobj.write('not actually python')
            clean = (workspace / 'clean.py').resolve()
            with clean.open('w') as fobj:
                fobj.write('print("hello")\n')
            self.invokeBlack([str(workspace)], exit_code=123)
            cache = black.read_cache(mode)
            self.assertNotIn(failing, cache)
            self.assertIn(clean, cache)

    def test_write_cache_write_fail(self):
        mode = DEFAULT_MODE
        with cache_dir(), patch.object(Path, 'open') as mock:
            mock.side_effect = OSError
            black.write_cache({}, [], mode)

    @event_loop()
    @patch('black.ProcessPoolExecutor', MagicMock(side_effect=OSError))
    def test_works_in_mono_process_only_environment(self):
        with cache_dir() as workspace:
            for f in [(workspace / 'one.py').resolve(), (workspace / 'two.py').resolve()]:
                f.write_text('print("hello")\n')
            self.invokeBlack([str(workspace)])

    @event_loop()
    def test_check_diff_use_together(self):
        with cache_dir():
            src1 = ((THIS_DIR / 'data') / 'string_quotes.py').resolve()
            self.invokeBlack([str(src1), '--diff', '--check'], exit_code=1)
            src2 = ((THIS_DIR / 'data') / 'composition.py').resolve()
            self.invokeBlack([str(src2), '--diff', '--check'])
            self.invokeBlack([str(src1), str(src2), '--diff', '--check'], exit_code=1)

    def test_no_files(self):
        with cache_dir():
            self.invokeBlack([])

    def test_broken_symlink(self):
        with cache_dir() as workspace:
            symlink = (workspace / 'broken_link.py')
            try:
                symlink.symlink_to('nonexistent.py')
            except OSError as e:
                self.skipTest(f"Can't create symlinks: {e}")
            self.invokeBlack([str(workspace.resolve())])

    def test_read_cache_line_lengths(self):
        mode = DEFAULT_MODE
        short_mode = replace(DEFAULT_MODE, line_length=1)
        with cache_dir() as workspace:
            path = (workspace / 'file.py').resolve()
            path.touch()
            black.write_cache({}, [path], mode)
            one = black.read_cache(mode)
            self.assertIn(path, one)
            two = black.read_cache(short_mode)
            self.assertNotIn(path, two)

    def test_single_file_force_pyi(self):
        pyi_mode = replace(DEFAULT_MODE, is_pyi=True)
        (contents, expected) = read_data('force_pyi')
        with cache_dir() as workspace:
            path = (workspace / 'file.py').resolve()
            with open(path, 'w') as fh:
                fh.write(contents)
            self.invokeBlack([str(path), '--pyi'])
            with open(path, 'r') as fh:
                actual = fh.read()
            pyi_cache = black.read_cache(pyi_mode)
            self.assertIn(path, pyi_cache)
            normal_cache = black.read_cache(DEFAULT_MODE)
            self.assertNotIn(path, normal_cache)
        self.assertFormatEqual(expected, actual)
        black.assert_equivalent(contents, actual)
        black.assert_stable(contents, actual, pyi_mode)

    @event_loop()
    def test_multi_file_force_pyi(self):
        reg_mode = DEFAULT_MODE
        pyi_mode = replace(DEFAULT_MODE, is_pyi=True)
        (contents, expected) = read_data('force_pyi')
        with cache_dir() as workspace:
            paths = [(workspace / 'file1.py').resolve(), (workspace / 'file2.py').resolve()]
            for path in paths:
                with open(path, 'w') as fh:
                    fh.write(contents)
            self.invokeBlack(([str(p) for p in paths] + ['--pyi']))
            for path in paths:
                with open(path, 'r') as fh:
                    actual = fh.read()
                self.assertEqual(actual, expected)
            pyi_cache = black.read_cache(pyi_mode)
            normal_cache = black.read_cache(reg_mode)
            for path in paths:
                self.assertIn(path, pyi_cache)
                self.assertNotIn(path, normal_cache)

    def test_pipe_force_pyi(self):
        (source, expected) = read_data('force_pyi')
        result = CliRunner().invoke(black.main, ['-', '-q', '--pyi'], input=BytesIO(source.encode('utf8')))
        self.assertEqual(result.exit_code, 0)
        actual = result.output
        self.assertFormatEqual(actual, expected)

    def test_single_file_force_py36(self):
        reg_mode = DEFAULT_MODE
        py36_mode = replace(DEFAULT_MODE, target_versions=PY36_VERSIONS)
        (source, expected) = read_data('force_py36')
        with cache_dir() as workspace:
            path = (workspace / 'file.py').resolve()
            with open(path, 'w') as fh:
                fh.write(source)
            self.invokeBlack([str(path), *PY36_ARGS])
            with open(path, 'r') as fh:
                actual = fh.read()
            py36_cache = black.read_cache(py36_mode)
            self.assertIn(path, py36_cache)
            normal_cache = black.read_cache(reg_mode)
            self.assertNotIn(path, normal_cache)
        self.assertEqual(actual, expected)

    @event_loop()
    def test_multi_file_force_py36(self):
        reg_mode = DEFAULT_MODE
        py36_mode = replace(DEFAULT_MODE, target_versions=PY36_VERSIONS)
        (source, expected) = read_data('force_py36')
        with cache_dir() as workspace:
            paths = [(workspace / 'file1.py').resolve(), (workspace / 'file2.py').resolve()]
            for path in paths:
                with open(path, 'w') as fh:
                    fh.write(source)
            self.invokeBlack(([str(p) for p in paths] + PY36_ARGS))
            for path in paths:
                with open(path, 'r') as fh:
                    actual = fh.read()
                self.assertEqual(actual, expected)
            pyi_cache = black.read_cache(py36_mode)
            normal_cache = black.read_cache(reg_mode)
            for path in paths:
                self.assertIn(path, pyi_cache)
                self.assertNotIn(path, normal_cache)

    def test_pipe_force_py36(self):
        (source, expected) = read_data('force_py36')
        result = CliRunner().invoke(black.main, ['-', '-q', '--target-version=py36'], input=BytesIO(source.encode('utf8')))
        self.assertEqual(result.exit_code, 0)
        actual = result.output
        self.assertFormatEqual(actual, expected)

    def test_include_exclude(self):
        path = ((THIS_DIR / 'data') / 'include_exclude_tests')
        include = re.compile('\\.pyi?$')
        exclude = re.compile('/exclude/|/\\.definitely_exclude/')
        report = black.Report()
        gitignore = PathSpec.from_lines('gitwildmatch', [])
        sources: List[Path] = []
        expected = [Path((path / 'b/dont_exclude/a.py')), Path((path / 'b/dont_exclude/a.pyi'))]
        this_abs = THIS_DIR.resolve()
        sources.extend(black.gen_python_files(path.iterdir(), this_abs, include, exclude, None, report, gitignore))
        self.assertEqual(sorted(expected), sorted(sources))

    @patch('black.find_project_root', (lambda *args: THIS_DIR.resolve()))
    def test_exclude_for_issue_1572(self):
        path = ((THIS_DIR / 'data') / 'include_exclude_tests')
        include = ''
        exclude = '/exclude/|a\\.py'
        src = str((path / 'b/exclude/a.py'))
        report = black.Report()
        expected = [Path((path / 'b/exclude/a.py'))]
        sources = list(black.get_sources(ctx=FakeContext(), src=(src,), quiet=True, verbose=False, include=include, exclude=exclude, force_exclude=None, report=report, stdin_filename=None))
        self.assertEqual(sorted(expected), sorted(sources))

    @patch('black.find_project_root', (lambda *args: THIS_DIR.resolve()))
    def test_get_sources_with_stdin(self):
        include = ''
        exclude = '/exclude/|a\\.py'
        src = '-'
        report = black.Report()
        expected = [Path('-')]
        sources = list(black.get_sources(ctx=FakeContext(), src=(src,), quiet=True, verbose=False, include=include, exclude=exclude, force_exclude=None, report=report, stdin_filename=None))
        self.assertEqual(sorted(expected), sorted(sources))

    @patch('black.find_project_root', (lambda *args: THIS_DIR.resolve()))
    def test_get_sources_with_stdin_filename(self):
        include = ''
        exclude = '/exclude/|a\\.py'
        src = '-'
        report = black.Report()
        stdin_filename = str((THIS_DIR / 'data/collections.py'))
        expected = [Path(f'__BLACK_STDIN_FILENAME__{stdin_filename}')]
        sources = list(black.get_sources(ctx=FakeContext(), src=(src,), quiet=True, verbose=False, include=include, exclude=exclude, force_exclude=None, report=report, stdin_filename=stdin_filename))
        self.assertEqual(sorted(expected), sorted(sources))

    @patch('black.find_project_root', (lambda *args: THIS_DIR.resolve()))
    def test_get_sources_with_stdin_filename_and_exclude(self):
        path = ((THIS_DIR / 'data') / 'include_exclude_tests')
        include = ''
        exclude = '/exclude/|a\\.py'
        src = '-'
        report = black.Report()
        stdin_filename = str((path / 'b/exclude/a.py'))
        expected = [Path(f'__BLACK_STDIN_FILENAME__{stdin_filename}')]
        sources = list(black.get_sources(ctx=FakeContext(), src=(src,), quiet=True, verbose=False, include=include, exclude=exclude, force_exclude=None, report=report, stdin_filename=stdin_filename))
        self.assertEqual(sorted(expected), sorted(sources))

    @patch('black.find_project_root', (lambda *args: THIS_DIR.resolve()))
    def test_get_sources_with_stdin_filename_and_force_exclude(self):
        path = ((THIS_DIR / 'data') / 'include_exclude_tests')
        include = ''
        force_exclude = '/exclude/|a\\.py'
        src = '-'
        report = black.Report()
        stdin_filename = str((path / 'b/exclude/a.py'))
        sources = list(black.get_sources(ctx=FakeContext(), src=(src,), quiet=True, verbose=False, include=include, exclude='', force_exclude=force_exclude, report=report, stdin_filename=stdin_filename))
        self.assertEqual([], sorted(sources))

    def test_reformat_one_with_stdin(self):
        with patch('black.format_stdin_to_stdout', return_value=(lambda *args, **kwargs: black.Changed.YES)) as fsts:
            report = MagicMock()
            path = Path('-')
            black.reformat_one(path, fast=True, write_back=black.WriteBack.YES, mode=DEFAULT_MODE, report=report)
            fsts.assert_called_once()
            report.done.assert_called_with(path, black.Changed.YES)

    def test_reformat_one_with_stdin_filename(self):
        with patch('black.format_stdin_to_stdout', return_value=(lambda *args, **kwargs: black.Changed.YES)) as fsts:
            report = MagicMock()
            p = 'foo.py'
            path = Path(f'__BLACK_STDIN_FILENAME__{p}')
            expected = Path(p)
            black.reformat_one(path, fast=True, write_back=black.WriteBack.YES, mode=DEFAULT_MODE, report=report)
            fsts.assert_called_once()
            report.done.assert_called_with(expected, black.Changed.YES)

    def test_reformat_one_with_stdin_and_existing_path(self):
        with patch('black.format_stdin_to_stdout', return_value=(lambda *args, **kwargs: black.Changed.YES)) as fsts:
            report = MagicMock()
            p = Path(str((THIS_DIR / 'data/collections.py')))
            self.assertTrue(p.is_file())
            path = Path(f'__BLACK_STDIN_FILENAME__{p}')
            expected = Path(p)
            black.reformat_one(path, fast=True, write_back=black.WriteBack.YES, mode=DEFAULT_MODE, report=report)
            fsts.assert_called_once()
            report.done.assert_called_with(expected, black.Changed.YES)

    def test_gitignore_exclude(self):
        path = ((THIS_DIR / 'data') / 'include_exclude_tests')
        include = re.compile('\\.pyi?$')
        exclude = re.compile('')
        report = black.Report()
        gitignore = PathSpec.from_lines('gitwildmatch', ['exclude/', '.definitely_exclude'])
        sources: List[Path] = []
        expected = [Path((path / 'b/dont_exclude/a.py')), Path((path / 'b/dont_exclude/a.pyi'))]
        this_abs = THIS_DIR.resolve()
        sources.extend(black.gen_python_files(path.iterdir(), this_abs, include, exclude, None, report, gitignore))
        self.assertEqual(sorted(expected), sorted(sources))

    def test_empty_include(self):
        path = ((THIS_DIR / 'data') / 'include_exclude_tests')
        report = black.Report()
        gitignore = PathSpec.from_lines('gitwildmatch', [])
        empty = re.compile('')
        sources: List[Path] = []
        expected = [Path((path / 'b/exclude/a.pie')), Path((path / 'b/exclude/a.py')), Path((path / 'b/exclude/a.pyi')), Path((path / 'b/dont_exclude/a.pie')), Path((path / 'b/dont_exclude/a.py')), Path((path / 'b/dont_exclude/a.pyi')), Path((path / 'b/.definitely_exclude/a.pie')), Path((path / 'b/.definitely_exclude/a.py')), Path((path / 'b/.definitely_exclude/a.pyi'))]
        this_abs = THIS_DIR.resolve()
        sources.extend(black.gen_python_files(path.iterdir(), this_abs, empty, re.compile(black.DEFAULT_EXCLUDES), None, report, gitignore))
        self.assertEqual(sorted(expected), sorted(sources))

    def test_empty_exclude(self):
        path = ((THIS_DIR / 'data') / 'include_exclude_tests')
        report = black.Report()
        gitignore = PathSpec.from_lines('gitwildmatch', [])
        empty = re.compile('')
        sources: List[Path] = []
        expected = [Path((path / 'b/dont_exclude/a.py')), Path((path / 'b/dont_exclude/a.pyi')), Path((path / 'b/exclude/a.py')), Path((path / 'b/exclude/a.pyi')), Path((path / 'b/.definitely_exclude/a.py')), Path((path / 'b/.definitely_exclude/a.pyi'))]
        this_abs = THIS_DIR.resolve()
        sources.extend(black.gen_python_files(path.iterdir(), this_abs, re.compile(black.DEFAULT_INCLUDES), empty, None, report, gitignore))
        self.assertEqual(sorted(expected), sorted(sources))

    def test_invalid_include_exclude(self):
        for option in ['--include', '--exclude']:
            self.invokeBlack(['-', option, '**()(!!*)'], exit_code=2)

    def test_preserves_line_endings(self):
        with TemporaryDirectory() as workspace:
            test_file = (Path(workspace) / 'test.py')
            for nl in ['\n', '\r\n']:
                contents = nl.join(['def f(  ):', '    pass'])
                test_file.write_bytes(contents.encode())
                ff(test_file, write_back=black.WriteBack.YES)
                updated_contents: bytes = test_file.read_bytes()
                self.assertIn(nl.encode(), updated_contents)
                if (nl == '\n'):
                    self.assertNotIn(b'\r\n', updated_contents)

    def test_preserves_line_endings_via_stdin(self):
        for nl in ['\n', '\r\n']:
            contents = nl.join(['def f(  ):', '    pass'])
            runner = BlackRunner()
            result = runner.invoke(black.main, ['-', '--fast'], input=BytesIO(contents.encode('utf8')))
            self.assertEqual(result.exit_code, 0)
            output = runner.stdout_bytes
            self.assertIn(nl.encode('utf8'), output)
            if (nl == '\n'):
                self.assertNotIn(b'\r\n', output)

    def test_assert_equivalent_different_asts(self):
        with self.assertRaises(AssertionError):
            black.assert_equivalent('{}', 'None')

    def test_symlink_out_of_root_directory(self):
        path = MagicMock()
        root = THIS_DIR.resolve()
        child = MagicMock()
        include = re.compile(black.DEFAULT_INCLUDES)
        exclude = re.compile(black.DEFAULT_EXCLUDES)
        report = black.Report()
        gitignore = PathSpec.from_lines('gitwildmatch', [])
        path.iterdir.return_value = [child]
        child.resolve.return_value = Path('/a/b/c')
        child.as_posix.return_value = '/a/b/c'
        child.is_symlink.return_value = True
        try:
            list(black.gen_python_files(path.iterdir(), root, include, exclude, None, report, gitignore))
        except ValueError as ve:
            self.fail(f'`get_python_files_in_dir()` failed: {ve}')
        path.iterdir.assert_called_once()
        child.resolve.assert_called_once()
        child.is_symlink.assert_called_once()
        child.is_symlink.return_value = False
        with self.assertRaises(ValueError):
            list(black.gen_python_files(path.iterdir(), root, include, exclude, None, report, gitignore))
        path.iterdir.assert_called()
        self.assertEqual(path.iterdir.call_count, 2)
        child.resolve.assert_called()
        self.assertEqual(child.resolve.call_count, 2)
        child.is_symlink.assert_called()
        self.assertEqual(child.is_symlink.call_count, 2)

    def test_shhh_click(self):
        try:
            from click import _unicodefun
        except ModuleNotFoundError:
            self.skipTest('Incompatible Click version')
        if (not hasattr(_unicodefun, '_verify_python3_env')):
            self.skipTest('Incompatible Click version')
        with patch('locale.getpreferredencoding') as gpe:
            gpe.return_value = 'ASCII'
            with self.assertRaises(RuntimeError):
                _unicodefun._verify_python3_env()
        black.patch_click()
        with patch('locale.getpreferredencoding') as gpe:
            gpe.return_value = 'ASCII'
            try:
                _unicodefun._verify_python3_env()
            except RuntimeError as re:
                self.fail(f'`patch_click()` failed, exception still raised: {re}')

    def test_root_logger_not_used_directly(self):

        def fail(*args: Any, **kwargs: Any) -> None:
            self.fail('Record created with root logger')
        with patch.multiple(logging.root, debug=fail, info=fail, warning=fail, error=fail, critical=fail, log=fail):
            ff(THIS_FILE)

    def test_invalid_config_return_code(self):
        tmp_file = Path(black.dump_to_file())
        try:
            tmp_config = Path(black.dump_to_file())
            tmp_config.unlink()
            args = ['--config', str(tmp_config), str(tmp_file)]
            self.invokeBlack(args, exit_code=2, ignore_config=False)
        finally:
            tmp_file.unlink()

    def test_parse_pyproject_toml(self):
        test_toml_file = (THIS_DIR / 'test.toml')
        config = black.parse_pyproject_toml(str(test_toml_file))
        self.assertEqual(config['verbose'], 1)
        self.assertEqual(config['check'], 'no')
        self.assertEqual(config['diff'], 'y')
        self.assertEqual(config['color'], True)
        self.assertEqual(config['line_length'], 79)
        self.assertEqual(config['target_version'], ['py36', 'py37', 'py38'])
        self.assertEqual(config['exclude'], '\\.pyi?$')
        self.assertEqual(config['include'], '\\.py?$')

    def test_read_pyproject_toml(self):
        test_toml_file = (THIS_DIR / 'test.toml')
        fake_ctx = FakeContext()
        black.read_pyproject_toml(fake_ctx, FakeParameter(), str(test_toml_file))
        config = fake_ctx.default_map
        self.assertEqual(config['verbose'], '1')
        self.assertEqual(config['check'], 'no')
        self.assertEqual(config['diff'], 'y')
        self.assertEqual(config['color'], 'True')
        self.assertEqual(config['line_length'], '79')
        self.assertEqual(config['target_version'], ['py36', 'py37', 'py38'])
        self.assertEqual(config['exclude'], '\\.pyi?$')
        self.assertEqual(config['include'], '\\.py?$')

    def test_find_project_root(self):
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            test_dir = (root / 'test')
            test_dir.mkdir()
            src_dir = (root / 'src')
            src_dir.mkdir()
            root_pyproject = (root / 'pyproject.toml')
            root_pyproject.touch()
            src_pyproject = (src_dir / 'pyproject.toml')
            src_pyproject.touch()
            src_python = (src_dir / 'foo.py')
            src_python.touch()
            self.assertEqual(black.find_project_root((src_dir, test_dir)), root.resolve())
            self.assertEqual(black.find_project_root((src_dir,)), src_dir.resolve())
            self.assertEqual(black.find_project_root((src_python,)), src_dir.resolve())

    def test_bpo_33660_workaround(self):
        if (system() == 'Windows'):
            return
        old_cwd = Path.cwd()
        try:
            root = Path('/')
            os.chdir(str(root))
            path = (Path('workspace') / 'project')
            report = black.Report(verbose=True)
            normalized_path = black.normalize_path_maybe_ignore(path, root, report)
            self.assertEqual(normalized_path, 'workspace/project')
        finally:
            os.chdir(str(old_cwd))
with open(black.__file__, 'r', encoding='utf-8') as _bf:
    black_source_lines = _bf.readlines()

def tracefunc(frame, event, arg):
    "Show function calls `from black/__init__.py` as they happen.\n\n    Register this with `sys.settrace()` in a test you're debugging.\n    "
    if (event != 'call'):
        return tracefunc
    stack = (len(inspect.stack()) - 19)
    stack *= 2
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno
    func_sig_lineno = (lineno - 1)
    funcname = black_source_lines[func_sig_lineno].strip()
    while funcname.startswith('@'):
        func_sig_lineno += 1
        funcname = black_source_lines[func_sig_lineno].strip()
    if ('black/__init__.py' in filename):
        print(f"{(' ' * stack)}{lineno}:{funcname}")
    return tracefunc
if (__name__ == '__main__'):
    unittest.main(module='test_black')
