
import ast
import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import Executor, ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from functools import lru_cache, partial, wraps
import io
import itertools
import logging
from multiprocessing import Manager, freeze_support
import os
from pathlib import Path
import pickle
import regex as re
import signal
import sys
import tempfile
import tokenize
import traceback
from typing import Any, Callable, Collection, Dict, Generator, Generic, Iterable, Iterator, List, Optional, Pattern, Sequence, Set, Sized, Tuple, Type, TypeVar, Union, cast, TYPE_CHECKING
from mypy_extensions import mypyc_attr
from appdirs import user_cache_dir
from dataclasses import dataclass, field, replace
import click
import toml
from typed_ast import ast3, ast27
from pathspec import PathSpec
from blib2to3.pytree import Node, Leaf, type_repr
from blib2to3 import pygram, pytree
from blib2to3.pgen2 import driver, token
from blib2to3.pgen2.grammar import Grammar
from blib2to3.pgen2.parse import ParseError
from _black_version import version as __version__
if (sys.version_info < (3, 8)):
    from typing_extensions import Final
else:
    from typing import Final
if TYPE_CHECKING:
    import colorama
DEFAULT_LINE_LENGTH = 88
DEFAULT_EXCLUDES = '/(\\.direnv|\\.eggs|\\.git|\\.hg|\\.mypy_cache|\\.nox|\\.tox|\\.venv|\\.svn|_build|buck-out|build|dist)/'
DEFAULT_INCLUDES = '\\.pyi?$'
CACHE_DIR = Path(user_cache_dir('black', version=__version__))
STDIN_PLACEHOLDER = '__BLACK_STDIN_FILENAME__'
STRING_PREFIX_CHARS = 'furbFURB'
FileContent = str
Encoding = str
NewLine = str
Depth = int
NodeType = int
ParserState = int
LeafID = int
StringID = int
Priority = int
Index = int
LN = Union[(Leaf, Node)]
Transformer = Callable[(['Line', Collection['Feature']], Iterator['Line'])]
Timestamp = float
FileSize = int
CacheInfo = Tuple[(Timestamp, FileSize)]
Cache = Dict[(Path, CacheInfo)]
out = partial(click.secho, bold=True, err=True)
err = partial(click.secho, fg='red', err=True)
pygram.initialize(CACHE_DIR)
syms = pygram.python_symbols

class NothingChanged(UserWarning):
    'Raised when reformatted code is the same as source.'

class CannotTransform(Exception):
    'Base class for errors raised by Transformers.'

class CannotSplit(CannotTransform):
    'A readable split that fits the allotted line length is impossible.'

class InvalidInput(ValueError):
    'Raised when input source code fails all parse attempts.'

class BracketMatchError(KeyError):
    'Raised when an opening bracket is unable to be matched to a closing bracket.'
T = TypeVar('T')
E = TypeVar('E', bound=Exception)

class Ok(Generic[T]):

    def __init__(self, value):
        self._value = value

    def ok(self):
        return self._value

class Err(Generic[E]):

    def __init__(self, e):
        self._e = e

    def err(self):
        return self._e
Result = Union[(Ok[T], Err[E])]
TResult = Result[(T, CannotTransform)]
TMatchResult = TResult[Index]

class WriteBack(Enum):
    NO = 0
    YES = 1
    DIFF = 2
    CHECK = 3
    COLOR_DIFF = 4

    @classmethod
    def from_configuration(cls, *, check, diff, color=False):
        if (check and (not diff)):
            return cls.CHECK
        if (diff and color):
            return cls.COLOR_DIFF
        return (cls.DIFF if diff else cls.YES)

class Changed(Enum):
    NO = 0
    CACHED = 1
    YES = 2

class TargetVersion(Enum):
    PY27 = 2
    PY33 = 3
    PY34 = 4
    PY35 = 5
    PY36 = 6
    PY37 = 7
    PY38 = 8
    PY39 = 9

    def is_python2(self):
        return (self is TargetVersion.PY27)

class Feature(Enum):
    UNICODE_LITERALS = 1
    F_STRINGS = 2
    NUMERIC_UNDERSCORES = 3
    TRAILING_COMMA_IN_CALL = 4
    TRAILING_COMMA_IN_DEF = 5
    ASYNC_IDENTIFIERS = 6
    ASYNC_KEYWORDS = 7
    ASSIGNMENT_EXPRESSIONS = 8
    POS_ONLY_ARGUMENTS = 9
    RELAXED_DECORATORS = 10
    FORCE_OPTIONAL_PARENTHESES = 50
VERSION_TO_FEATURES = {TargetVersion.PY27: {Feature.ASYNC_IDENTIFIERS}, TargetVersion.PY33: {Feature.UNICODE_LITERALS, Feature.ASYNC_IDENTIFIERS}, TargetVersion.PY34: {Feature.UNICODE_LITERALS, Feature.ASYNC_IDENTIFIERS}, TargetVersion.PY35: {Feature.UNICODE_LITERALS, Feature.TRAILING_COMMA_IN_CALL, Feature.ASYNC_IDENTIFIERS}, TargetVersion.PY36: {Feature.UNICODE_LITERALS, Feature.F_STRINGS, Feature.NUMERIC_UNDERSCORES, Feature.TRAILING_COMMA_IN_CALL, Feature.TRAILING_COMMA_IN_DEF, Feature.ASYNC_IDENTIFIERS}, TargetVersion.PY37: {Feature.UNICODE_LITERALS, Feature.F_STRINGS, Feature.NUMERIC_UNDERSCORES, Feature.TRAILING_COMMA_IN_CALL, Feature.TRAILING_COMMA_IN_DEF, Feature.ASYNC_KEYWORDS}, TargetVersion.PY38: {Feature.UNICODE_LITERALS, Feature.F_STRINGS, Feature.NUMERIC_UNDERSCORES, Feature.TRAILING_COMMA_IN_CALL, Feature.TRAILING_COMMA_IN_DEF, Feature.ASYNC_KEYWORDS, Feature.ASSIGNMENT_EXPRESSIONS, Feature.POS_ONLY_ARGUMENTS}, TargetVersion.PY39: {Feature.UNICODE_LITERALS, Feature.F_STRINGS, Feature.NUMERIC_UNDERSCORES, Feature.TRAILING_COMMA_IN_CALL, Feature.TRAILING_COMMA_IN_DEF, Feature.ASYNC_KEYWORDS, Feature.ASSIGNMENT_EXPRESSIONS, Feature.RELAXED_DECORATORS, Feature.POS_ONLY_ARGUMENTS}}

@dataclass
class Mode():
    target_versions = field(default_factory=set)
    line_length = DEFAULT_LINE_LENGTH
    string_normalization = True
    experimental_string_processing = False
    is_pyi = False

    def get_cache_key(self):
        if self.target_versions:
            version_str = ','.join((str(version.value) for version in sorted(self.target_versions, key=(lambda v: v.value))))
        else:
            version_str = '-'
        parts = [version_str, str(self.line_length), str(int(self.string_normalization)), str(int(self.is_pyi))]
        return '.'.join(parts)
FileMode = Mode

def supports_feature(target_versions, feature):
    return all(((feature in VERSION_TO_FEATURES[version]) for version in target_versions))

def find_pyproject_toml(path_search_start):
    'Find the absolute filepath to a pyproject.toml if it exists'
    path_project_root = find_project_root(path_search_start)
    path_pyproject_toml = (path_project_root / 'pyproject.toml')
    return (str(path_pyproject_toml) if path_pyproject_toml.is_file() else None)

def parse_pyproject_toml(path_config):
    'Parse a pyproject toml file, pulling out relevant parts for Black\n\n    If parsing fails, will raise a toml.TomlDecodeError\n    '
    pyproject_toml = toml.load(path_config)
    config = pyproject_toml.get('tool', {}).get('black', {})
    return {k.replace('--', '').replace('-', '_'): v for (k, v) in config.items()}

def read_pyproject_toml(ctx, param, value):
    'Inject Black configuration from "pyproject.toml" into defaults in `ctx`.\n\n    Returns the path to a successfully found and read configuration file, None\n    otherwise.\n    '
    if (not value):
        value = find_pyproject_toml(ctx.params.get('src', ()))
        if (value is None):
            return None
    try:
        config = parse_pyproject_toml(value)
    except (toml.TomlDecodeError, OSError) as e:
        raise click.FileError(filename=value, hint=f'Error reading configuration file: {e}')
    if (not config):
        return None
    else:
        config = {k: (str(v) if (not isinstance(v, (list, dict))) else v) for (k, v) in config.items()}
    target_version = config.get('target_version')
    if ((target_version is not None) and (not isinstance(target_version, list))):
        raise click.BadOptionUsage('target-version', 'Config key target-version must be a list')
    default_map: Dict[(str, Any)] = {}
    if ctx.default_map:
        default_map.update(ctx.default_map)
    default_map.update(config)
    ctx.default_map = default_map
    return value

def target_version_option_callback(c, p, v):
    "Compute the target versions from a --target-version flag.\n\n    This is its own function because mypy couldn't infer the type correctly\n    when it was a lambda, causing mypyc trouble.\n    "
    return [TargetVersion[val.upper()] for val in v]

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('-c', '--code', type=str, help='Format the code passed in as a string.')
@click.option('-l', '--line-length', type=int, default=DEFAULT_LINE_LENGTH, help='How many characters per line to allow.', show_default=True)
@click.option('-t', '--target-version', type=click.Choice([v.name.lower() for v in TargetVersion]), callback=target_version_option_callback, multiple=True, help="Python versions that should be supported by Black's output. [default: per-file auto-detection]")
@click.option('--pyi', is_flag=True, help='Format all input files like typing stubs regardless of file extension (useful when piping source on standard input).')
@click.option('-S', '--skip-string-normalization', is_flag=True, help="Don't normalize string quotes or prefixes.")
@click.option('--experimental-string-processing', is_flag=True, hidden=True, help='Experimental option that performs more normalization on string literals. Currently disabled because it leads to some crashes.')
@click.option('--check', is_flag=True, help="Don't write the files back, just return the status.  Return code 0 means nothing would change.  Return code 1 means some files would be reformatted. Return code 123 means there was an internal error.")
@click.option('--diff', is_flag=True, help="Don't write the files back, just output a diff for each file on stdout.")
@click.option('--color/--no-color', is_flag=True, help='Show colored diff. Only applies when `--diff` is given.')
@click.option('--fast/--safe', is_flag=True, help='If --fast given, skip temporary sanity checks. [default: --safe]')
@click.option('--include', type=str, default=DEFAULT_INCLUDES, help='A regular expression that matches files and directories that should be included on recursive searches.  An empty value means all files are included regardless of the name.  Use forward slashes for directories on all platforms (Windows, too).  Exclusions are calculated first, inclusions later.', show_default=True)
@click.option('--exclude', type=str, default=DEFAULT_EXCLUDES, help='A regular expression that matches files and directories that should be excluded on recursive searches.  An empty value means no paths are excluded. Use forward slashes for directories on all platforms (Windows, too).  Exclusions are calculated first, inclusions later.', show_default=True)
@click.option('--force-exclude', type=str, help='Like --exclude, but files and directories matching this regex will be excluded even when they are passed explicitly as arguments.')
@click.option('--stdin-filename', type=str, help='The name of the file when passing it through stdin. Useful to make sure Black will respect --force-exclude option on some editors that rely on using stdin.')
@click.option('-q', '--quiet', is_flag=True, help="Don't emit non-error messages to stderr. Errors are still emitted; silence those with 2>/dev/null.")
@click.option('-v', '--verbose', is_flag=True, help='Also emit messages to stderr about files that were not changed or were ignored due to --exclude=.')
@click.version_option(version=__version__)
@click.argument('src', nargs=(- 1), type=click.Path(exists=True, file_okay=True, dir_okay=True, readable=True, allow_dash=True), is_eager=True)
@click.option('--config', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, allow_dash=False, path_type=str), is_eager=True, callback=read_pyproject_toml, help='Read configuration from FILE path.')
@click.pass_context
def main(ctx, code, line_length, target_version, check, diff, color, fast, pyi, skip_string_normalization, experimental_string_processing, quiet, verbose, include, exclude, force_exclude, stdin_filename, src, config):
    'The uncompromising code formatter.'
    write_back = WriteBack.from_configuration(check=check, diff=diff, color=color)
    if target_version:
        versions = set(target_version)
    else:
        versions = set()
    mode = Mode(target_versions=versions, line_length=line_length, is_pyi=pyi, string_normalization=(not skip_string_normalization), experimental_string_processing=experimental_string_processing)
    if (config and verbose):
        out(f'Using configuration from {config}.', bold=False, fg='blue')
    if (code is not None):
        print(format_str(code, mode=mode))
        ctx.exit(0)
    report = Report(check=check, diff=diff, quiet=quiet, verbose=verbose)
    sources = get_sources(ctx=ctx, src=src, quiet=quiet, verbose=verbose, include=include, exclude=exclude, force_exclude=force_exclude, report=report, stdin_filename=stdin_filename)
    path_empty(sources, 'No Python files are present to be formatted. Nothing to do 😴', quiet, verbose, ctx)
    if (len(sources) == 1):
        reformat_one(src=sources.pop(), fast=fast, write_back=write_back, mode=mode, report=report)
    else:
        reformat_many(sources=sources, fast=fast, write_back=write_back, mode=mode, report=report)
    if (verbose or (not quiet)):
        out(('Oh no! 💥 💔 💥' if report.return_code else 'All done! ✨ 🍰 ✨'))
        click.secho(str(report), err=True)
    ctx.exit(report.return_code)

def get_sources(*, ctx, src, quiet, verbose, include, exclude, force_exclude, report, stdin_filename):
    'Compute the set of files to be formatted.'
    try:
        include_regex = re_compile_maybe_verbose(include)
    except re.error:
        err(f'Invalid regular expression for include given: {include!r}')
        ctx.exit(2)
    try:
        exclude_regex = re_compile_maybe_verbose(exclude)
    except re.error:
        err(f'Invalid regular expression for exclude given: {exclude!r}')
        ctx.exit(2)
    try:
        force_exclude_regex = (re_compile_maybe_verbose(force_exclude) if force_exclude else None)
    except re.error:
        err(f'Invalid regular expression for force_exclude given: {force_exclude!r}')
        ctx.exit(2)
    root = find_project_root(src)
    sources: Set[Path] = set()
    path_empty(src, 'No Path provided. Nothing to do 😴', quiet, verbose, ctx)
    gitignore = get_gitignore(root)
    for s in src:
        if ((s == '-') and stdin_filename):
            p = Path(stdin_filename)
            is_stdin = True
        else:
            p = Path(s)
            is_stdin = False
        if (is_stdin or p.is_file()):
            normalized_path = normalize_path_maybe_ignore(p, root, report)
            if (normalized_path is None):
                continue
            normalized_path = ('/' + normalized_path)
            if force_exclude_regex:
                force_exclude_match = force_exclude_regex.search(normalized_path)
            else:
                force_exclude_match = None
            if (force_exclude_match and force_exclude_match.group(0)):
                report.path_ignored(p, 'matches the --force-exclude regular expression')
                continue
            if is_stdin:
                p = Path(f'{STDIN_PLACEHOLDER}{str(p)}')
            sources.add(p)
        elif p.is_dir():
            sources.update(gen_python_files(p.iterdir(), root, include_regex, exclude_regex, force_exclude_regex, report, gitignore))
        elif (s == '-'):
            sources.add(p)
        else:
            err(f'invalid path: {s}')
    return sources

def path_empty(src, msg, quiet, verbose, ctx):
    '\n    Exit if there is no `src` provided for formatting\n    '
    if ((not src) and (verbose or (not quiet))):
        out(msg)
        ctx.exit(0)

def reformat_one(src, fast, write_back, mode, report):
    'Reformat a single file under `src` without spawning child processes.\n\n    `fast`, `write_back`, and `mode` options are passed to\n    :func:`format_file_in_place` or :func:`format_stdin_to_stdout`.\n    '
    try:
        changed = Changed.NO
        if (str(src) == '-'):
            is_stdin = True
        elif str(src).startswith(STDIN_PLACEHOLDER):
            is_stdin = True
            src = Path(str(src)[len(STDIN_PLACEHOLDER):])
        else:
            is_stdin = False
        if is_stdin:
            if format_stdin_to_stdout(fast=fast, write_back=write_back, mode=mode):
                changed = Changed.YES
        else:
            cache: Cache = {}
            if (write_back not in (WriteBack.DIFF, WriteBack.COLOR_DIFF)):
                cache = read_cache(mode)
                res_src = src.resolve()
                if ((res_src in cache) and (cache[res_src] == get_cache_info(res_src))):
                    changed = Changed.CACHED
            if ((changed is not Changed.CACHED) and format_file_in_place(src, fast=fast, write_back=write_back, mode=mode)):
                changed = Changed.YES
            if (((write_back is WriteBack.YES) and (changed is not Changed.CACHED)) or ((write_back is WriteBack.CHECK) and (changed is Changed.NO))):
                write_cache(cache, [src], mode)
        report.done(src, changed)
    except Exception as exc:
        if report.verbose:
            traceback.print_exc()
        report.failed(src, str(exc))

def reformat_many(sources, fast, write_back, mode, report):
    'Reformat multiple files using a ProcessPoolExecutor.'
    executor: Executor
    loop = asyncio.get_event_loop()
    worker_count = os.cpu_count()
    if (sys.platform == 'win32'):
        worker_count = min(worker_count, 60)
    try:
        executor = ProcessPoolExecutor(max_workers=worker_count)
    except (ImportError, OSError):
        executor = ThreadPoolExecutor(max_workers=1)
    try:
        loop.run_until_complete(schedule_formatting(sources=sources, fast=fast, write_back=write_back, mode=mode, report=report, loop=loop, executor=executor))
    finally:
        shutdown(loop)
        if (executor is not None):
            executor.shutdown()

async def schedule_formatting(sources, fast, write_back, mode, report, loop, executor):
    'Run formatting of `sources` in parallel using the provided `executor`.\n\n    (Use ProcessPoolExecutors for actual parallelism.)\n\n    `write_back`, `fast`, and `mode` options are passed to\n    :func:`format_file_in_place`.\n    '
    cache: Cache = {}
    if (write_back not in (WriteBack.DIFF, WriteBack.COLOR_DIFF)):
        cache = read_cache(mode)
        (sources, cached) = filter_cached(cache, sources)
        for src in sorted(cached):
            report.done(src, Changed.CACHED)
    if (not sources):
        return
    cancelled = []
    sources_to_cache = []
    lock = None
    if (write_back in (WriteBack.DIFF, WriteBack.COLOR_DIFF)):
        manager = Manager()
        lock = manager.Lock()
    tasks = {asyncio.ensure_future(loop.run_in_executor(executor, format_file_in_place, src, fast, mode, write_back, lock)): src for src in sorted(sources)}
    pending: Iterable['asyncio.Future[bool]'] = tasks.keys()
    try:
        loop.add_signal_handler(signal.SIGINT, cancel, pending)
        loop.add_signal_handler(signal.SIGTERM, cancel, pending)
    except NotImplementedError:
        pass
    while pending:
        (done, _) = (await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED))
        for task in done:
            src = tasks.pop(task)
            if task.cancelled():
                cancelled.append(task)
            elif task.exception():
                report.failed(src, str(task.exception()))
            else:
                changed = (Changed.YES if task.result() else Changed.NO)
                if ((write_back is WriteBack.YES) or ((write_back is WriteBack.CHECK) and (changed is Changed.NO))):
                    sources_to_cache.append(src)
                report.done(src, changed)
    if cancelled:
        (await asyncio.gather(*cancelled, loop=loop, return_exceptions=True))
    if sources_to_cache:
        write_cache(cache, sources_to_cache, mode)

def format_file_in_place(src, fast, mode, write_back=WriteBack.NO, lock=None):
    'Format file under `src` path. Return True if changed.\n\n    If `write_back` is DIFF, write a diff to stdout. If it is YES, write reformatted\n    code to the file.\n    `mode` and `fast` options are passed to :func:`format_file_contents`.\n    '
    if (src.suffix == '.pyi'):
        mode = replace(mode, is_pyi=True)
    then = datetime.utcfromtimestamp(src.stat().st_mtime)
    with open(src, 'rb') as buf:
        (src_contents, encoding, newline) = decode_bytes(buf.read())
    try:
        dst_contents = format_file_contents(src_contents, fast=fast, mode=mode)
    except NothingChanged:
        return False
    if (write_back == WriteBack.YES):
        with open(src, 'w', encoding=encoding, newline=newline) as f:
            f.write(dst_contents)
    elif (write_back in (WriteBack.DIFF, WriteBack.COLOR_DIFF)):
        now = datetime.utcnow()
        src_name = f'{src}	{then} +0000'
        dst_name = f'{src}	{now} +0000'
        diff_contents = diff(src_contents, dst_contents, src_name, dst_name)
        if (write_back == write_back.COLOR_DIFF):
            diff_contents = color_diff(diff_contents)
        with (lock or nullcontext()):
            f = io.TextIOWrapper(sys.stdout.buffer, encoding=encoding, newline=newline, write_through=True)
            f = wrap_stream_for_windows(f)
            f.write(diff_contents)
            f.detach()
    return True

def color_diff(contents):
    'Inject the ANSI color codes to the diff.'
    lines = contents.split('\n')
    for (i, line) in enumerate(lines):
        if (line.startswith('+++') or line.startswith('---')):
            line = (('\x1b[1;37m' + line) + '\x1b[0m')
        elif line.startswith('@@'):
            line = (('\x1b[36m' + line) + '\x1b[0m')
        elif line.startswith('+'):
            line = (('\x1b[32m' + line) + '\x1b[0m')
        elif line.startswith('-'):
            line = (('\x1b[31m' + line) + '\x1b[0m')
        lines[i] = line
    return '\n'.join(lines)

def wrap_stream_for_windows(f):
    "\n    Wrap stream with colorama's wrap_stream so colors are shown on Windows.\n\n    If `colorama` is unavailable, the original stream is returned unmodified.\n    Otherwise, the `wrap_stream()` function determines whether the stream needs\n    to be wrapped for a Windows environment and will accordingly either return\n    an `AnsiToWin32` wrapper or the original stream.\n    "
    try:
        from colorama.initialise import wrap_stream
    except ImportError:
        return f
    else:
        return wrap_stream(f, convert=None, strip=False, autoreset=False, wrap=True)

def format_stdin_to_stdout(fast, *, write_back=WriteBack.NO, mode):
    'Format file on stdin. Return True if changed.\n\n    If `write_back` is YES, write reformatted code back to stdout. If it is DIFF,\n    write a diff to stdout. The `mode` argument is passed to\n    :func:`format_file_contents`.\n    '
    then = datetime.utcnow()
    (src, encoding, newline) = decode_bytes(sys.stdin.buffer.read())
    dst = src
    try:
        dst = format_file_contents(src, fast=fast, mode=mode)
        return True
    except NothingChanged:
        return False
    finally:
        f = io.TextIOWrapper(sys.stdout.buffer, encoding=encoding, newline=newline, write_through=True)
        if (write_back == WriteBack.YES):
            f.write(dst)
        elif (write_back in (WriteBack.DIFF, WriteBack.COLOR_DIFF)):
            now = datetime.utcnow()
            src_name = f'STDIN	{then} +0000'
            dst_name = f'STDOUT	{now} +0000'
            d = diff(src, dst, src_name, dst_name)
            if (write_back == WriteBack.COLOR_DIFF):
                d = color_diff(d)
                f = wrap_stream_for_windows(f)
            f.write(d)
        f.detach()

def format_file_contents(src_contents, *, fast, mode):
    'Reformat contents of a file and return new contents.\n\n    If `fast` is False, additionally confirm that the reformatted code is\n    valid by calling :func:`assert_equivalent` and :func:`assert_stable` on it.\n    `mode` is passed to :func:`format_str`.\n    '
    if (not src_contents.strip()):
        raise NothingChanged
    dst_contents = format_str(src_contents, mode=mode)
    if (src_contents == dst_contents):
        raise NothingChanged
    if (not fast):
        assert_equivalent(src_contents, dst_contents)
        assert_stable(src_contents, dst_contents, mode=mode)
    return dst_contents

def format_str(src_contents, *, mode):
    'Reformat a string and return new contents.\n\n    `mode` determines formatting options, such as how many characters per line are\n    allowed.  Example:\n\n    >>> import black\n    >>> print(black.format_str("def f(arg:str=\'\')->None:...", mode=black.Mode()))\n    def f(arg: str = "") -> None:\n        ...\n\n    A more complex example:\n\n    >>> print(\n    ...   black.format_str(\n    ...     "def f(arg:str=\'\')->None: hey",\n    ...     mode=black.Mode(\n    ...       target_versions={black.TargetVersion.PY36},\n    ...       line_length=10,\n    ...       string_normalization=False,\n    ...       is_pyi=False,\n    ...     ),\n    ...   ),\n    ... )\n    def f(\n        arg: str = \'\',\n    ) -> None:\n        hey\n\n    '
    src_node = lib2to3_parse(src_contents.lstrip(), mode.target_versions)
    dst_contents = []
    future_imports = get_future_imports(src_node)
    if mode.target_versions:
        versions = mode.target_versions
    else:
        versions = detect_target_versions(src_node)
    normalize_fmt_off(src_node)
    lines = LineGenerator(remove_u_prefix=(('unicode_literals' in future_imports) or supports_feature(versions, Feature.UNICODE_LITERALS)), is_pyi=mode.is_pyi, normalize_strings=mode.string_normalization)
    elt = EmptyLineTracker(is_pyi=mode.is_pyi)
    empty_line = Line()
    after = 0
    split_line_features = {feature for feature in {Feature.TRAILING_COMMA_IN_CALL, Feature.TRAILING_COMMA_IN_DEF} if supports_feature(versions, feature)}
    for current_line in lines.visit(src_node):
        dst_contents.append((str(empty_line) * after))
        (before, after) = elt.maybe_empty_lines(current_line)
        dst_contents.append((str(empty_line) * before))
        for line in transform_line(current_line, mode=mode, features=split_line_features):
            dst_contents.append(str(line))
    return ''.join(dst_contents)

def decode_bytes(src):
    'Return a tuple of (decoded_contents, encoding, newline).\n\n    `newline` is either CRLF or LF but `decoded_contents` is decoded with\n    universal newlines (i.e. only contains LF).\n    '
    srcbuf = io.BytesIO(src)
    (encoding, lines) = tokenize.detect_encoding(srcbuf.readline)
    if (not lines):
        return ('', encoding, '\n')
    newline = ('\r\n' if (b'\r\n' == lines[0][(- 2):]) else '\n')
    srcbuf.seek(0)
    with io.TextIOWrapper(srcbuf, encoding) as tiow:
        return (tiow.read(), encoding, newline)

def get_grammars(target_versions):
    if (not target_versions):
        return [pygram.python_grammar_no_print_statement_no_exec_statement_async_keywords, pygram.python_grammar_no_print_statement_no_exec_statement, pygram.python_grammar_no_print_statement, pygram.python_grammar]
    if all((version.is_python2() for version in target_versions)):
        return [pygram.python_grammar_no_print_statement, pygram.python_grammar]
    grammars = []
    if (not supports_feature(target_versions, Feature.ASYNC_IDENTIFIERS)):
        grammars.append(pygram.python_grammar_no_print_statement_no_exec_statement_async_keywords)
    if (not supports_feature(target_versions, Feature.ASYNC_KEYWORDS)):
        grammars.append(pygram.python_grammar_no_print_statement_no_exec_statement)
    return grammars

def lib2to3_parse(src_txt, target_versions=()):
    'Given a string with source, return the lib2to3 Node.'
    if (not src_txt.endswith('\n')):
        src_txt += '\n'
    for grammar in get_grammars(set(target_versions)):
        drv = driver.Driver(grammar, pytree.convert)
        try:
            result = drv.parse_string(src_txt, True)
            break
        except ParseError as pe:
            (lineno, column) = pe.context[1]
            lines = src_txt.splitlines()
            try:
                faulty_line = lines[(lineno - 1)]
            except IndexError:
                faulty_line = '<line number missing in source>'
            exc = InvalidInput(f'Cannot parse: {lineno}:{column}: {faulty_line}')
    else:
        raise exc from None
    if isinstance(result, Leaf):
        result = Node(syms.file_input, [result])
    return result

def lib2to3_unparse(node):
    'Given a lib2to3 node, return its string representation.'
    code = str(node)
    return code

class Visitor(Generic[T]):
    'Basic lib2to3 visitor that yields things of type `T` on `visit()`.'

    def visit(self, node):
        'Main method to visit `node` and its children.\n\n        It tries to find a `visit_*()` method for the given `node.type`, like\n        `visit_simple_stmt` for Node objects or `visit_INDENT` for Leaf objects.\n        If no dedicated `visit_*()` method is found, chooses `visit_default()`\n        instead.\n\n        Then yields objects of type `T` from the selected visitor.\n        '
        if (node.type < 256):
            name = token.tok_name[node.type]
        else:
            name = str(type_repr(node.type))
        visitf = getattr(self, f'visit_{name}', None)
        if visitf:
            (yield from visitf(node))
        else:
            (yield from self.visit_default(node))

    def visit_default(self, node):
        'Default `visit_*()` implementation. Recurses to children of `node`.'
        if isinstance(node, Node):
            for child in node.children:
                (yield from self.visit(child))

@dataclass
class DebugVisitor(Visitor[T]):
    tree_depth = 0

    def visit_default(self, node):
        indent = (' ' * (2 * self.tree_depth))
        if isinstance(node, Node):
            _type = type_repr(node.type)
            out(f'{indent}{_type}', fg='yellow')
            self.tree_depth += 1
            for child in node.children:
                (yield from self.visit(child))
            self.tree_depth -= 1
            out(f'{indent}/{_type}', fg='yellow', bold=False)
        else:
            _type = token.tok_name.get(node.type, str(node.type))
            out(f'{indent}{_type}', fg='blue', nl=False)
            if node.prefix:
                out(f' {node.prefix!r}', fg='green', bold=False, nl=False)
            out(f' {node.value!r}', fg='blue', bold=False)

    @classmethod
    def show(cls, code):
        'Pretty-print the lib2to3 AST of a given string of `code`.\n\n        Convenience method for debugging.\n        '
        v: DebugVisitor[None] = DebugVisitor()
        if isinstance(code, str):
            code = lib2to3_parse(code)
        list(v.visit(code))
WHITESPACE = {token.DEDENT, token.INDENT, token.NEWLINE}
STATEMENT = {syms.if_stmt, syms.while_stmt, syms.for_stmt, syms.try_stmt, syms.except_clause, syms.with_stmt, syms.funcdef, syms.classdef}
STANDALONE_COMMENT = 153
token.tok_name[STANDALONE_COMMENT] = 'STANDALONE_COMMENT'
LOGIC_OPERATORS = {'and', 'or'}
COMPARATORS = {token.LESS, token.GREATER, token.EQEQUAL, token.NOTEQUAL, token.LESSEQUAL, token.GREATEREQUAL}
MATH_OPERATORS = {token.VBAR, token.CIRCUMFLEX, token.AMPER, token.LEFTSHIFT, token.RIGHTSHIFT, token.PLUS, token.MINUS, token.STAR, token.SLASH, token.DOUBLESLASH, token.PERCENT, token.AT, token.TILDE, token.DOUBLESTAR}
STARS = {token.STAR, token.DOUBLESTAR}
VARARGS_SPECIALS = (STARS | {token.SLASH})
VARARGS_PARENTS = {syms.arglist, syms.argument, syms.trailer, syms.typedargslist, syms.varargslist}
UNPACKING_PARENTS = {syms.atom, syms.dictsetmaker, syms.listmaker, syms.testlist_gexp, syms.testlist_star_expr}
TEST_DESCENDANTS = {syms.test, syms.lambdef, syms.or_test, syms.and_test, syms.not_test, syms.comparison, syms.star_expr, syms.expr, syms.xor_expr, syms.and_expr, syms.shift_expr, syms.arith_expr, syms.trailer, syms.term, syms.power}
ASSIGNMENTS = {'=', '+=', '-=', '*=', '@=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=', '**=', '//='}
COMPREHENSION_PRIORITY = 20
COMMA_PRIORITY = 18
TERNARY_PRIORITY = 16
LOGIC_PRIORITY = 14
STRING_PRIORITY = 12
COMPARATOR_PRIORITY = 10
MATH_PRIORITIES = {token.VBAR: 9, token.CIRCUMFLEX: 8, token.AMPER: 7, token.LEFTSHIFT: 6, token.RIGHTSHIFT: 6, token.PLUS: 5, token.MINUS: 5, token.STAR: 4, token.SLASH: 4, token.DOUBLESLASH: 4, token.PERCENT: 4, token.AT: 4, token.TILDE: 3, token.DOUBLESTAR: 2}
DOT_PRIORITY = 1

@dataclass
class BracketTracker():
    'Keeps track of brackets on a line.'
    depth = 0
    bracket_match = field(default_factory=dict)
    delimiters = field(default_factory=dict)
    previous = None
    _for_loop_depths = field(default_factory=list)
    _lambda_argument_depths = field(default_factory=list)
    invisible = field(default_factory=list)

    def mark(self, leaf):
        "Mark `leaf` with bracket-related metadata. Keep track of delimiters.\n\n        All leaves receive an int `bracket_depth` field that stores how deep\n        within brackets a given leaf is. 0 means there are no enclosing brackets\n        that started on this line.\n\n        If a leaf is itself a closing bracket, it receives an `opening_bracket`\n        field that it forms a pair with. This is a one-directional link to\n        avoid reference cycles.\n\n        If a leaf is a delimiter (a token on which Black can split the line if\n        needed) and it's on depth 0, its `id()` is stored in the tracker's\n        `delimiters` field.\n        "
        if (leaf.type == token.COMMENT):
            return
        self.maybe_decrement_after_for_loop_variable(leaf)
        self.maybe_decrement_after_lambda_arguments(leaf)
        if (leaf.type in CLOSING_BRACKETS):
            self.depth -= 1
            try:
                opening_bracket = self.bracket_match.pop((self.depth, leaf.type))
            except KeyError as e:
                raise BracketMatchError(f'Unable to match a closing bracket to the following opening bracket: {leaf}') from e
            leaf.opening_bracket = opening_bracket
            if (not leaf.value):
                self.invisible.append(leaf)
        leaf.bracket_depth = self.depth
        if (self.depth == 0):
            delim = is_split_before_delimiter(leaf, self.previous)
            if (delim and (self.previous is not None)):
                self.delimiters[id(self.previous)] = delim
            else:
                delim = is_split_after_delimiter(leaf, self.previous)
                if delim:
                    self.delimiters[id(leaf)] = delim
        if (leaf.type in OPENING_BRACKETS):
            self.bracket_match[(self.depth, BRACKET[leaf.type])] = leaf
            self.depth += 1
            if (not leaf.value):
                self.invisible.append(leaf)
        self.previous = leaf
        self.maybe_increment_lambda_arguments(leaf)
        self.maybe_increment_for_loop_variable(leaf)

    def any_open_brackets(self):
        'Return True if there is an yet unmatched open bracket on the line.'
        return bool(self.bracket_match)

    def max_delimiter_priority(self, exclude=()):
        'Return the highest priority of a delimiter found on the line.\n\n        Values are consistent with what `is_split_*_delimiter()` return.\n        Raises ValueError on no delimiters.\n        '
        return max((v for (k, v) in self.delimiters.items() if (k not in exclude)))

    def delimiter_count_with_priority(self, priority=0):
        'Return the number of delimiters with the given `priority`.\n\n        If no `priority` is passed, defaults to max priority on the line.\n        '
        if (not self.delimiters):
            return 0
        priority = (priority or self.max_delimiter_priority())
        return sum((1 for p in self.delimiters.values() if (p == priority)))

    def maybe_increment_for_loop_variable(self, leaf):
        'In a for loop, or comprehension, the variables are often unpacks.\n\n        To avoid splitting on the comma in this situation, increase the depth of\n        tokens between `for` and `in`.\n        '
        if ((leaf.type == token.NAME) and (leaf.value == 'for')):
            self.depth += 1
            self._for_loop_depths.append(self.depth)
            return True
        return False

    def maybe_decrement_after_for_loop_variable(self, leaf):
        'See `maybe_increment_for_loop_variable` above for explanation.'
        if (self._for_loop_depths and (self._for_loop_depths[(- 1)] == self.depth) and (leaf.type == token.NAME) and (leaf.value == 'in')):
            self.depth -= 1
            self._for_loop_depths.pop()
            return True
        return False

    def maybe_increment_lambda_arguments(self, leaf):
        'In a lambda expression, there might be more than one argument.\n\n        To avoid splitting on the comma in this situation, increase the depth of\n        tokens between `lambda` and `:`.\n        '
        if ((leaf.type == token.NAME) and (leaf.value == 'lambda')):
            self.depth += 1
            self._lambda_argument_depths.append(self.depth)
            return True
        return False

    def maybe_decrement_after_lambda_arguments(self, leaf):
        'See `maybe_increment_lambda_arguments` above for explanation.'
        if (self._lambda_argument_depths and (self._lambda_argument_depths[(- 1)] == self.depth) and (leaf.type == token.COLON)):
            self.depth -= 1
            self._lambda_argument_depths.pop()
            return True
        return False

    def get_open_lsqb(self):
        'Return the most recent opening square bracket (if any).'
        return self.bracket_match.get(((self.depth - 1), token.RSQB))

@dataclass
class Line():
    'Holds leaves and comments. Can be printed with `str(line)`.'
    depth = 0
    leaves = field(default_factory=list)
    comments = field(default_factory=dict)
    bracket_tracker = field(default_factory=BracketTracker)
    inside_brackets = False
    should_explode = False

    def append(self, leaf, preformatted=False):
        'Add a new `leaf` to the end of the line.\n\n        Unless `preformatted` is True, the `leaf` will receive a new consistent\n        whitespace prefix and metadata applied by :class:`BracketTracker`.\n        Trailing commas are maybe removed, unpacked for loop variables are\n        demoted from being delimiters.\n\n        Inline comments are put aside.\n        '
        has_value = ((leaf.type in BRACKETS) or bool(leaf.value.strip()))
        if (not has_value):
            return
        if ((token.COLON == leaf.type) and self.is_class_paren_empty):
            del self.leaves[(- 2):]
        if (self.leaves and (not preformatted)):
            leaf.prefix += whitespace(leaf, complex_subscript=self.is_complex_subscript(leaf))
        if (self.inside_brackets or (not preformatted)):
            self.bracket_tracker.mark(leaf)
            if self.maybe_should_explode(leaf):
                self.should_explode = True
        if (not self.append_comment(leaf)):
            self.leaves.append(leaf)

    def append_safe(self, leaf, preformatted=False):
        'Like :func:`append()` but disallow invalid standalone comment structure.\n\n        Raises ValueError when any `leaf` is appended after a standalone comment\n        or when a standalone comment is not the first leaf on the line.\n        '
        if (self.bracket_tracker.depth == 0):
            if self.is_comment:
                raise ValueError('cannot append to standalone comments')
            if (self.leaves and (leaf.type == STANDALONE_COMMENT)):
                raise ValueError('cannot append standalone comments to a populated line')
        self.append(leaf, preformatted=preformatted)

    @property
    def is_comment(self):
        'Is this line a standalone comment?'
        return ((len(self.leaves) == 1) and (self.leaves[0].type == STANDALONE_COMMENT))

    @property
    def is_decorator(self):
        'Is this line a decorator?'
        return (bool(self) and (self.leaves[0].type == token.AT))

    @property
    def is_import(self):
        'Is this an import line?'
        return (bool(self) and is_import(self.leaves[0]))

    @property
    def is_class(self):
        'Is this line a class definition?'
        return (bool(self) and (self.leaves[0].type == token.NAME) and (self.leaves[0].value == 'class'))

    @property
    def is_stub_class(self):
        'Is this line a class definition with a body consisting only of "..."?'
        return (self.is_class and (self.leaves[(- 3):] == [Leaf(token.DOT, '.') for _ in range(3)]))

    @property
    def is_def(self):
        'Is this a function definition? (Also returns True for async defs.)'
        try:
            first_leaf = self.leaves[0]
        except IndexError:
            return False
        try:
            second_leaf: Optional[Leaf] = self.leaves[1]
        except IndexError:
            second_leaf = None
        return (((first_leaf.type == token.NAME) and (first_leaf.value == 'def')) or ((first_leaf.type == token.ASYNC) and (second_leaf is not None) and (second_leaf.type == token.NAME) and (second_leaf.value == 'def')))

    @property
    def is_class_paren_empty(self):
        'Is this a class with no base classes but using parentheses?\n\n        Those are unnecessary and should be removed.\n        '
        return (bool(self) and (len(self.leaves) == 4) and self.is_class and (self.leaves[2].type == token.LPAR) and (self.leaves[2].value == '(') and (self.leaves[3].type == token.RPAR) and (self.leaves[3].value == ')'))

    @property
    def is_triple_quoted_string(self):
        'Is the line a triple quoted string?'
        return (bool(self) and (self.leaves[0].type == token.STRING) and self.leaves[0].value.startswith(('"""', "'''")))

    def contains_standalone_comments(self, depth_limit=sys.maxsize):
        'If so, needs to be split before emitting.'
        for leaf in self.leaves:
            if ((leaf.type == STANDALONE_COMMENT) and (leaf.bracket_depth <= depth_limit)):
                return True
        return False

    def contains_uncollapsable_type_comments(self):
        ignored_ids = set()
        try:
            last_leaf = self.leaves[(- 1)]
            ignored_ids.add(id(last_leaf))
            if ((last_leaf.type == token.COMMA) or ((last_leaf.type == token.RPAR) and (not last_leaf.value))):
                last_leaf = self.leaves[(- 2)]
                ignored_ids.add(id(last_leaf))
        except IndexError:
            return False
        comment_seen = False
        for (leaf_id, comments) in self.comments.items():
            for comment in comments:
                if is_type_comment(comment):
                    if (comment_seen or ((not is_type_comment(comment, ' ignore')) and (leaf_id not in ignored_ids))):
                        return True
                comment_seen = True
        return False

    def contains_unsplittable_type_ignore(self):
        if (not self.leaves):
            return False
        first_line = next((leaf.lineno for leaf in self.leaves if (leaf.lineno != 0)), 0)
        last_line = next((leaf.lineno for leaf in reversed(self.leaves) if (leaf.lineno != 0)), 0)
        if (first_line == last_line):
            for node in self.leaves[(- 2):]:
                for comment in self.comments.get(id(node), []):
                    if is_type_comment(comment, ' ignore'):
                        return True
        return False

    def contains_multiline_strings(self):
        return any((is_multiline_string(leaf) for leaf in self.leaves))

    def maybe_should_explode(self, closing):
        "Return True if this line should explode (always be split), that is when:\n        - there's a trailing comma here; and\n        - it's not a one-tuple.\n        "
        if (not ((closing.type in CLOSING_BRACKETS) and self.leaves and (self.leaves[(- 1)].type == token.COMMA))):
            return False
        if (closing.type in {token.RBRACE, token.RSQB}):
            return True
        if self.is_import:
            return True
        if (not is_one_tuple_between(closing.opening_bracket, closing, self.leaves)):
            return True
        return False

    def append_comment(self, comment):
        'Add an inline or standalone comment to the line.'
        if ((comment.type == STANDALONE_COMMENT) and self.bracket_tracker.any_open_brackets()):
            comment.prefix = ''
            return False
        if (comment.type != token.COMMENT):
            return False
        if (not self.leaves):
            comment.type = STANDALONE_COMMENT
            comment.prefix = ''
            return False
        last_leaf = self.leaves[(- 1)]
        if ((last_leaf.type == token.RPAR) and (not last_leaf.value) and last_leaf.parent and (len(list(last_leaf.parent.leaves())) <= 3) and (not is_type_comment(comment))):
            if (len(self.leaves) < 2):
                comment.type = STANDALONE_COMMENT
                comment.prefix = ''
                return False
            last_leaf = self.leaves[(- 2)]
        self.comments.setdefault(id(last_leaf), []).append(comment)
        return True

    def comments_after(self, leaf):
        'Generate comments that should appear directly after `leaf`.'
        return self.comments.get(id(leaf), [])

    def remove_trailing_comma(self):
        'Remove the trailing comma and moves the comments attached to it.'
        trailing_comma = self.leaves.pop()
        trailing_comma_comments = self.comments.pop(id(trailing_comma), [])
        self.comments.setdefault(id(self.leaves[(- 1)]), []).extend(trailing_comma_comments)

    def is_complex_subscript(self, leaf):
        'Return True iff `leaf` is part of a slice with non-trivial exprs.'
        open_lsqb = self.bracket_tracker.get_open_lsqb()
        if (open_lsqb is None):
            return False
        subscript_start = open_lsqb.next_sibling
        if isinstance(subscript_start, Node):
            if (subscript_start.type == syms.listmaker):
                return False
            if (subscript_start.type == syms.subscriptlist):
                subscript_start = child_towards(subscript_start, leaf)
        return ((subscript_start is not None) and any(((n.type in TEST_DESCENDANTS) for n in subscript_start.pre_order())))

    def clone(self):
        return Line(depth=self.depth, inside_brackets=self.inside_brackets, should_explode=self.should_explode)

    def __str__(self):
        'Render the line.'
        if (not self):
            return '\n'
        indent = ('    ' * self.depth)
        leaves = iter(self.leaves)
        first = next(leaves)
        res = f'{first.prefix}{indent}{first.value}'
        for leaf in leaves:
            res += str(leaf)
        for comment in itertools.chain.from_iterable(self.comments.values()):
            res += str(comment)
        return (res + '\n')

    def __bool__(self):
        'Return True if the line has leaves or comments.'
        return bool((self.leaves or self.comments))

@dataclass
class EmptyLineTracker():
    "Provides a stateful method that returns the number of potential extra\n    empty lines needed before and after the currently processed line.\n\n    Note: this tracker works on lines that haven't been split yet.  It assumes\n    the prefix of the first leaf consists of optional newlines.  Those newlines\n    are consumed by `maybe_empty_lines()` and included in the computation.\n    "
    is_pyi = False
    previous_line = None
    previous_after = 0
    previous_defs = field(default_factory=list)

    def maybe_empty_lines(self, current_line):
        'Return the number of extra empty lines before and after the `current_line`.\n\n        This is for separating `def`, `async def` and `class` with extra empty\n        lines (two on module-level).\n        '
        (before, after) = self._maybe_empty_lines(current_line)
        before = (0 if (self.previous_line is None) else (before - self.previous_after))
        self.previous_after = after
        self.previous_line = current_line
        return (before, after)

    def _maybe_empty_lines(self, current_line):
        max_allowed = 1
        if (current_line.depth == 0):
            max_allowed = (1 if self.is_pyi else 2)
        if current_line.leaves:
            first_leaf = current_line.leaves[0]
            before = first_leaf.prefix.count('\n')
            before = min(before, max_allowed)
            first_leaf.prefix = ''
        else:
            before = 0
        depth = current_line.depth
        while (self.previous_defs and (self.previous_defs[(- 1)] >= depth)):
            self.previous_defs.pop()
            if self.is_pyi:
                before = (0 if depth else 1)
            else:
                before = (1 if depth else 2)
        if (current_line.is_decorator or current_line.is_def or current_line.is_class):
            return self._maybe_empty_lines_for_class_or_def(current_line, before)
        if (self.previous_line and self.previous_line.is_import and (not current_line.is_import) and (depth == self.previous_line.depth)):
            return ((before or 1), 0)
        if (self.previous_line and self.previous_line.is_class and current_line.is_triple_quoted_string):
            return (before, 1)
        return (before, 0)

    def _maybe_empty_lines_for_class_or_def(self, current_line, before):
        if (not current_line.is_decorator):
            self.previous_defs.append(current_line.depth)
        if (self.previous_line is None):
            return (0, 0)
        if self.previous_line.is_decorator:
            if (self.is_pyi and current_line.is_stub_class):
                return (0, 1)
            return (0, 0)
        if ((self.previous_line.depth < current_line.depth) and (self.previous_line.is_class or self.previous_line.is_def)):
            return (0, 0)
        if (self.previous_line.is_comment and (self.previous_line.depth == current_line.depth) and (before == 0)):
            return (0, 0)
        if self.is_pyi:
            if (self.previous_line.depth > current_line.depth):
                newlines = 1
            elif (current_line.is_class or self.previous_line.is_class):
                if (current_line.is_stub_class and self.previous_line.is_stub_class):
                    newlines = 0
                else:
                    newlines = 1
            elif ((current_line.is_def or current_line.is_decorator) and (not self.previous_line.is_def)):
                newlines = 1
            else:
                newlines = 0
        else:
            newlines = 2
        if (current_line.depth and newlines):
            newlines -= 1
        return (newlines, 0)

@dataclass
class LineGenerator(Visitor[Line]):
    "Generates reformatted Line objects.  Empty lines are not emitted.\n\n    Note: destroys the tree it's visiting by mutating prefixes of its leaves\n    in ways that will no longer stringify to valid Python code on the tree.\n    "
    is_pyi = False
    normalize_strings = True
    current_line = field(default_factory=Line)
    remove_u_prefix = False

    def line(self, indent=0):
        'Generate a line.\n\n        If the line is empty, only emit if it makes sense.\n        If the line is too long, split it first and then generate.\n\n        If any lines were generated, set up a new current_line.\n        '
        if (not self.current_line):
            self.current_line.depth += indent
            return
        complete_line = self.current_line
        self.current_line = Line(depth=(complete_line.depth + indent))
        (yield complete_line)

    def visit_default(self, node):
        'Default `visit_*()` implementation. Recurses to children of `node`.'
        if isinstance(node, Leaf):
            any_open_brackets = self.current_line.bracket_tracker.any_open_brackets()
            for comment in generate_comments(node):
                if any_open_brackets:
                    self.current_line.append(comment)
                elif (comment.type == token.COMMENT):
                    self.current_line.append(comment)
                    (yield from self.line())
                else:
                    (yield from self.line())
                    self.current_line.append(comment)
                    (yield from self.line())
            normalize_prefix(node, inside_brackets=any_open_brackets)
            if (self.normalize_strings and (node.type == token.STRING)):
                normalize_string_prefix(node, remove_u_prefix=self.remove_u_prefix)
                normalize_string_quotes(node)
            if (node.type == token.NUMBER):
                normalize_numeric_literal(node)
            if (node.type not in WHITESPACE):
                self.current_line.append(node)
        (yield from super().visit_default(node))

    def visit_INDENT(self, node):
        'Increase indentation level, maybe yield a line.'
        (yield from self.line((+ 1)))
        (yield from self.visit_default(node))

    def visit_DEDENT(self, node):
        'Decrease indentation level, maybe yield a line.'
        (yield from self.line())
        (yield from self.visit_default(node))
        (yield from self.line((- 1)))

    def visit_stmt(self, node, keywords, parens):
        'Visit a statement.\n\n        This implementation is shared for `if`, `while`, `for`, `try`, `except`,\n        `def`, `with`, `class`, `assert` and assignments.\n\n        The relevant Python language `keywords` for a given statement will be\n        NAME leaves within it. This methods puts those on a separate line.\n\n        `parens` holds a set of string leaf values immediately after which\n        invisible parens should be put.\n        '
        normalize_invisible_parens(node, parens_after=parens)
        for child in node.children:
            if ((child.type == token.NAME) and (child.value in keywords)):
                (yield from self.line())
            (yield from self.visit(child))

    def visit_suite(self, node):
        'Visit a suite.'
        if (self.is_pyi and is_stub_suite(node)):
            (yield from self.visit(node.children[2]))
        else:
            (yield from self.visit_default(node))

    def visit_simple_stmt(self, node):
        'Visit a statement without nested statements.'
        is_suite_like = (node.parent and (node.parent.type in STATEMENT))
        if is_suite_like:
            if (self.is_pyi and is_stub_body(node)):
                (yield from self.visit_default(node))
            else:
                (yield from self.line((+ 1)))
                (yield from self.visit_default(node))
                (yield from self.line((- 1)))
        else:
            if ((not self.is_pyi) or (not node.parent) or (not is_stub_suite(node.parent))):
                (yield from self.line())
            (yield from self.visit_default(node))

    def visit_async_stmt(self, node):
        'Visit `async def`, `async for`, `async with`.'
        (yield from self.line())
        children = iter(node.children)
        for child in children:
            (yield from self.visit(child))
            if (child.type == token.ASYNC):
                break
        internal_stmt = next(children)
        for child in internal_stmt.children:
            (yield from self.visit(child))

    def visit_decorators(self, node):
        'Visit decorators.'
        for child in node.children:
            (yield from self.line())
            (yield from self.visit(child))

    def visit_SEMI(self, leaf):
        'Remove a semicolon and put the other statement on a separate line.'
        (yield from self.line())

    def visit_ENDMARKER(self, leaf):
        'End of file. Process outstanding comments and end with a newline.'
        (yield from self.visit_default(leaf))
        (yield from self.line())

    def visit_STANDALONE_COMMENT(self, leaf):
        if (not self.current_line.bracket_tracker.any_open_brackets()):
            (yield from self.line())
        (yield from self.visit_default(leaf))

    def visit_factor(self, node):
        'Force parentheses between a unary op and a binary power:\n\n        -2 ** 8 -> -(2 ** 8)\n        '
        (_operator, operand) = node.children
        if ((operand.type == syms.power) and (len(operand.children) == 3) and (operand.children[1].type == token.DOUBLESTAR)):
            lpar = Leaf(token.LPAR, '(')
            rpar = Leaf(token.RPAR, ')')
            index = (operand.remove() or 0)
            node.insert_child(index, Node(syms.atom, [lpar, operand, rpar]))
        (yield from self.visit_default(node))

    def visit_STRING(self, leaf):
        if (is_docstring(leaf) and ('\\\n' not in leaf.value)):
            prefix = get_string_prefix(leaf.value)
            lead_len = (len(prefix) + 3)
            tail_len = (- 3)
            indent = ((' ' * 4) * self.current_line.depth)
            docstring = fix_docstring(leaf.value[lead_len:tail_len], indent)
            if docstring:
                if (leaf.value[(lead_len - 1)] == docstring[0]):
                    docstring = (' ' + docstring)
                if (leaf.value[(tail_len + 1)] == docstring[(- 1)]):
                    docstring = (docstring + ' ')
            leaf.value = ((leaf.value[0:lead_len] + docstring) + leaf.value[tail_len:])
        (yield from self.visit_default(leaf))

    def __post_init__(self):
        'You are in a twisty little maze of passages.'
        v = self.visit_stmt
        Ø: Set[str] = set()
        self.visit_assert_stmt = partial(v, keywords={'assert'}, parens={'assert', ','})
        self.visit_if_stmt = partial(v, keywords={'if', 'else', 'elif'}, parens={'if', 'elif'})
        self.visit_while_stmt = partial(v, keywords={'while', 'else'}, parens={'while'})
        self.visit_for_stmt = partial(v, keywords={'for', 'else'}, parens={'for', 'in'})
        self.visit_try_stmt = partial(v, keywords={'try', 'except', 'else', 'finally'}, parens=Ø)
        self.visit_except_clause = partial(v, keywords={'except'}, parens=Ø)
        self.visit_with_stmt = partial(v, keywords={'with'}, parens=Ø)
        self.visit_funcdef = partial(v, keywords={'def'}, parens=Ø)
        self.visit_classdef = partial(v, keywords={'class'}, parens=Ø)
        self.visit_expr_stmt = partial(v, keywords=Ø, parens=ASSIGNMENTS)
        self.visit_return_stmt = partial(v, keywords={'return'}, parens={'return'})
        self.visit_import_from = partial(v, keywords=Ø, parens={'import'})
        self.visit_del_stmt = partial(v, keywords=Ø, parens={'del'})
        self.visit_async_funcdef = self.visit_async_stmt
        self.visit_decorated = self.visit_decorators
IMPLICIT_TUPLE = {syms.testlist, syms.testlist_star_expr, syms.exprlist}
BRACKET = {token.LPAR: token.RPAR, token.LSQB: token.RSQB, token.LBRACE: token.RBRACE}
OPENING_BRACKETS = set(BRACKET.keys())
CLOSING_BRACKETS = set(BRACKET.values())
BRACKETS = (OPENING_BRACKETS | CLOSING_BRACKETS)
ALWAYS_NO_SPACE = (CLOSING_BRACKETS | {token.COMMA, STANDALONE_COMMENT})

def whitespace(leaf, *, complex_subscript):
    'Return whitespace prefix if needed for the given `leaf`.\n\n    `complex_subscript` signals whether the given leaf is part of a subscription\n    which has non-trivial arguments, like arithmetic expressions or function calls.\n    '
    NO = ''
    SPACE = ' '
    DOUBLESPACE = '  '
    t = leaf.type
    p = leaf.parent
    v = leaf.value
    if (t in ALWAYS_NO_SPACE):
        return NO
    if (t == token.COMMENT):
        return DOUBLESPACE
    assert (p is not None), f'INTERNAL ERROR: hand-made leaf without parent: {leaf!r}'
    if ((t == token.COLON) and (p.type not in {syms.subscript, syms.subscriptlist, syms.sliceop})):
        return NO
    prev = leaf.prev_sibling
    if (not prev):
        prevp = preceding_leaf(p)
        if ((not prevp) or (prevp.type in OPENING_BRACKETS)):
            return NO
        if (t == token.COLON):
            if (prevp.type == token.COLON):
                return NO
            elif ((prevp.type != token.COMMA) and (not complex_subscript)):
                return NO
            return SPACE
        if (prevp.type == token.EQUAL):
            if prevp.parent:
                if (prevp.parent.type in {syms.arglist, syms.argument, syms.parameters, syms.varargslist}):
                    return NO
                elif (prevp.parent.type == syms.typedargslist):
                    return prevp.prefix
        elif (prevp.type in VARARGS_SPECIALS):
            if is_vararg(prevp, within=(VARARGS_PARENTS | UNPACKING_PARENTS)):
                return NO
        elif (prevp.type == token.COLON):
            if (prevp.parent and (prevp.parent.type in {syms.subscript, syms.sliceop})):
                return (SPACE if complex_subscript else NO)
        elif (prevp.parent and (prevp.parent.type == syms.factor) and (prevp.type in MATH_OPERATORS)):
            return NO
        elif ((prevp.type == token.RIGHTSHIFT) and prevp.parent and (prevp.parent.type == syms.shift_expr) and prevp.prev_sibling and (prevp.prev_sibling.type == token.NAME) and (prevp.prev_sibling.value == 'print')):
            return NO
        elif ((prevp.type == token.AT) and p.parent and (p.parent.type == syms.decorator)):
            return NO
    elif (prev.type in OPENING_BRACKETS):
        return NO
    if (p.type in {syms.parameters, syms.arglist}):
        if ((not prev) or (prev.type != token.COMMA)):
            return NO
    elif (p.type == syms.varargslist):
        if (prev and (prev.type != token.COMMA)):
            return NO
    elif (p.type == syms.typedargslist):
        if (not prev):
            return NO
        if (t == token.EQUAL):
            if (prev.type != syms.tname):
                return NO
        elif (prev.type == token.EQUAL):
            return prev.prefix
        elif (prev.type != token.COMMA):
            return NO
    elif (p.type == syms.tname):
        if (not prev):
            prevp = preceding_leaf(p)
            if ((not prevp) or (prevp.type != token.COMMA)):
                return NO
    elif (p.type == syms.trailer):
        if ((t == token.LPAR) or (t == token.RPAR)):
            return NO
        if (not prev):
            if (t == token.DOT):
                prevp = preceding_leaf(p)
                if ((not prevp) or (prevp.type != token.NUMBER)):
                    return NO
            elif (t == token.LSQB):
                return NO
        elif (prev.type != token.COMMA):
            return NO
    elif (p.type == syms.argument):
        if (t == token.EQUAL):
            return NO
        if (not prev):
            prevp = preceding_leaf(p)
            if ((not prevp) or (prevp.type == token.LPAR)):
                return NO
        elif (prev.type in ({token.EQUAL} | VARARGS_SPECIALS)):
            return NO
    elif (p.type == syms.decorator):
        return NO
    elif (p.type == syms.dotted_name):
        if prev:
            return NO
        prevp = preceding_leaf(p)
        if ((not prevp) or (prevp.type == token.AT) or (prevp.type == token.DOT)):
            return NO
    elif (p.type == syms.classdef):
        if (t == token.LPAR):
            return NO
        if (prev and (prev.type == token.LPAR)):
            return NO
    elif (p.type in {syms.subscript, syms.sliceop}):
        if (not prev):
            assert (p.parent is not None), 'subscripts are always parented'
            if (p.parent.type == syms.subscriptlist):
                return SPACE
            return NO
        elif (not complex_subscript):
            return NO
    elif (p.type == syms.atom):
        if (prev and (t == token.DOT)):
            return NO
    elif (p.type == syms.dictsetmaker):
        if (prev and (prev.type == token.DOUBLESTAR)):
            return NO
    elif (p.type in {syms.factor, syms.star_expr}):
        if (not prev):
            prevp = preceding_leaf(p)
            if ((not prevp) or (prevp.type in OPENING_BRACKETS)):
                return NO
            prevp_parent = prevp.parent
            assert (prevp_parent is not None)
            if ((prevp.type == token.COLON) and (prevp_parent.type in {syms.subscript, syms.sliceop})):
                return NO
            elif ((prevp.type == token.EQUAL) and (prevp_parent.type == syms.argument)):
                return NO
        elif (t in {token.NAME, token.NUMBER, token.STRING}):
            return NO
    elif (p.type == syms.import_from):
        if (t == token.DOT):
            if (prev and (prev.type == token.DOT)):
                return NO
        elif (t == token.NAME):
            if (v == 'import'):
                return SPACE
            if (prev and (prev.type == token.DOT)):
                return NO
    elif (p.type == syms.sliceop):
        return NO
    return SPACE

def preceding_leaf(node):
    'Return the first leaf that precedes `node`, if any.'
    while node:
        res = node.prev_sibling
        if res:
            if isinstance(res, Leaf):
                return res
            try:
                return list(res.leaves())[(- 1)]
            except IndexError:
                return None
        node = node.parent
    return None

def prev_siblings_are(node, tokens):
    "Return if the `node` and its previous siblings match types against the provided\n    list of tokens; the provided `node`has its type matched against the last element in\n    the list.  `None` can be used as the first element to declare that the start of the\n    list is anchored at the start of its parent's children."
    if (not tokens):
        return True
    if (tokens[(- 1)] is None):
        return (node is None)
    if (not node):
        return False
    if (node.type != tokens[(- 1)]):
        return False
    return prev_siblings_are(node.prev_sibling, tokens[:(- 1)])

def child_towards(ancestor, descendant):
    'Return the child of `ancestor` that contains `descendant`.'
    node: Optional[LN] = descendant
    while (node and (node.parent != ancestor)):
        node = node.parent
    return node

def container_of(leaf):
    'Return `leaf` or one of its ancestors that is the topmost container of it.\n\n    By "container" we mean a node where `leaf` is the very first child.\n    '
    same_prefix = leaf.prefix
    container: LN = leaf
    while container:
        parent = container.parent
        if (parent is None):
            break
        if (parent.children[0].prefix != same_prefix):
            break
        if (parent.type == syms.file_input):
            break
        if ((parent.prev_sibling is not None) and (parent.prev_sibling.type in BRACKETS)):
            break
        container = parent
    return container

def is_split_after_delimiter(leaf, previous=None):
    'Return the priority of the `leaf` delimiter, given a line break after it.\n\n    The delimiter priorities returned here are from those delimiters that would\n    cause a line break after themselves.\n\n    Higher numbers are higher priority.\n    '
    if (leaf.type == token.COMMA):
        return COMMA_PRIORITY
    return 0

def is_split_before_delimiter(leaf, previous=None):
    'Return the priority of the `leaf` delimiter, given a line break before it.\n\n    The delimiter priorities returned here are from those delimiters that would\n    cause a line break before themselves.\n\n    Higher numbers are higher priority.\n    '
    if is_vararg(leaf, within=(VARARGS_PARENTS | UNPACKING_PARENTS)):
        return 0
    if ((leaf.type == token.DOT) and leaf.parent and (leaf.parent.type not in {syms.import_from, syms.dotted_name}) and ((previous is None) or (previous.type in CLOSING_BRACKETS))):
        return DOT_PRIORITY
    if ((leaf.type in MATH_OPERATORS) and leaf.parent and (leaf.parent.type not in {syms.factor, syms.star_expr})):
        return MATH_PRIORITIES[leaf.type]
    if (leaf.type in COMPARATORS):
        return COMPARATOR_PRIORITY
    if ((leaf.type == token.STRING) and (previous is not None) and (previous.type == token.STRING)):
        return STRING_PRIORITY
    if (leaf.type not in {token.NAME, token.ASYNC}):
        return 0
    if (((leaf.value == 'for') and leaf.parent and (leaf.parent.type in {syms.comp_for, syms.old_comp_for})) or (leaf.type == token.ASYNC)):
        if ((not isinstance(leaf.prev_sibling, Leaf)) or (leaf.prev_sibling.value != 'async')):
            return COMPREHENSION_PRIORITY
    if ((leaf.value == 'if') and leaf.parent and (leaf.parent.type in {syms.comp_if, syms.old_comp_if})):
        return COMPREHENSION_PRIORITY
    if ((leaf.value in {'if', 'else'}) and leaf.parent and (leaf.parent.type == syms.test)):
        return TERNARY_PRIORITY
    if (leaf.value == 'is'):
        return COMPARATOR_PRIORITY
    if ((leaf.value == 'in') and leaf.parent and (leaf.parent.type in {syms.comp_op, syms.comparison}) and (not ((previous is not None) and (previous.type == token.NAME) and (previous.value == 'not')))):
        return COMPARATOR_PRIORITY
    if ((leaf.value == 'not') and leaf.parent and (leaf.parent.type == syms.comp_op) and (not ((previous is not None) and (previous.type == token.NAME) and (previous.value == 'is')))):
        return COMPARATOR_PRIORITY
    if ((leaf.value in LOGIC_OPERATORS) and leaf.parent):
        return LOGIC_PRIORITY
    return 0
FMT_OFF = {'# fmt: off', '# fmt:off', '# yapf: disable'}
FMT_ON = {'# fmt: on', '# fmt:on', '# yapf: enable'}

def generate_comments(leaf):
    'Clean the prefix of the `leaf` and generate comments from it, if any.\n\n    Comments in lib2to3 are shoved into the whitespace prefix.  This happens\n    in `pgen2/driver.py:Driver.parse_tokens()`.  This was a brilliant implementation\n    move because it does away with modifying the grammar to include all the\n    possible places in which comments can be placed.\n\n    The sad consequence for us though is that comments don\'t "belong" anywhere.\n    This is why this function generates simple parentless Leaf objects for\n    comments.  We simply don\'t know what the correct parent should be.\n\n    No matter though, we can live without this.  We really only need to\n    differentiate between inline and standalone comments.  The latter don\'t\n    share the line with any code.\n\n    Inline comments are emitted as regular token.COMMENT leaves.  Standalone\n    are emitted with a fake STANDALONE_COMMENT token identifier.\n    '
    for pc in list_comments(leaf.prefix, is_endmarker=(leaf.type == token.ENDMARKER)):
        (yield Leaf(pc.type, pc.value, prefix=('\n' * pc.newlines)))

@dataclass
class ProtoComment():
    "Describes a piece of syntax that is a comment.\n\n    It's not a :class:`blib2to3.pytree.Leaf` so that:\n\n    * it can be cached (`Leaf` objects should not be reused more than once as\n      they store their lineno, column, prefix, and parent information);\n    * `newlines` and `consumed` fields are kept separate from the `value`. This\n      simplifies handling of special marker comments like ``# fmt: off/on``.\n    "

@lru_cache(maxsize=4096)
def list_comments(prefix, *, is_endmarker):
    'Return a list of :class:`ProtoComment` objects parsed from the given `prefix`.'
    result: List[ProtoComment] = []
    if ((not prefix) or ('#' not in prefix)):
        return result
    consumed = 0
    nlines = 0
    ignored_lines = 0
    for (index, line) in enumerate(prefix.split('\n')):
        consumed += (len(line) + 1)
        line = line.lstrip()
        if (not line):
            nlines += 1
        if (not line.startswith('#')):
            if line.endswith('\\'):
                ignored_lines += 1
            continue
        if ((index == ignored_lines) and (not is_endmarker)):
            comment_type = token.COMMENT
        else:
            comment_type = STANDALONE_COMMENT
        comment = make_comment(line)
        result.append(ProtoComment(type=comment_type, value=comment, newlines=nlines, consumed=consumed))
        nlines = 0
    return result

def make_comment(content):
    'Return a consistently formatted comment from the given `content` string.\n\n    All comments (except for "##", "#!", "#:", \'#\'", "#%%") should have a single\n    space between the hash sign and the content.\n\n    If `content` didn\'t start with a hash sign, one is provided.\n    '
    content = content.rstrip()
    if (not content):
        return '#'
    if (content[0] == '#'):
        content = content[1:]
    if (content and (content[0] not in " !:#'%")):
        content = (' ' + content)
    return ('#' + content)

def transform_line(line, mode, features=()):
    'Transform a `line`, potentially splitting it into many lines.\n\n    They should fit in the allotted `line_length` but might not be able to.\n\n    `features` are syntactical features that may be used in the output.\n    '
    if line.is_comment:
        (yield line)
        return
    line_str = line_to_string(line)

    def init_st(ST: Type[StringTransformer]) -> StringTransformer:
        'Initialize StringTransformer'
        return ST(mode.line_length, mode.string_normalization)
    string_merge = init_st(StringMerger)
    string_paren_strip = init_st(StringParenStripper)
    string_split = init_st(StringSplitter)
    string_paren_wrap = init_st(StringParenWrapper)
    transformers: List[Transformer]
    if ((not line.contains_uncollapsable_type_comments()) and (not line.should_explode) and (is_line_short_enough(line, line_length=mode.line_length, line_str=line_str) or line.contains_unsplittable_type_ignore()) and (not (line.inside_brackets and line.contains_standalone_comments()))):
        if mode.experimental_string_processing:
            transformers = [string_merge, string_paren_strip]
        else:
            transformers = []
    elif line.is_def:
        transformers = [left_hand_split]
    else:

        def rhs(line: Line, features: Collection[Feature]) -> Iterator[Line]:
            'Wraps calls to `right_hand_split`.\n\n            The calls increasingly `omit` right-hand trailers (bracket pairs with\n            content), meaning the trailers get glued together to split on another\n            bracket pair instead.\n            '
            for omit in generate_trailers_to_omit(line, mode.line_length):
                lines = list(right_hand_split(line, mode.line_length, features, omit=omit))
                if is_line_short_enough(lines[0], line_length=mode.line_length):
                    (yield from lines)
                    return
            (yield from right_hand_split(line, line_length=mode.line_length, features=features))
        if mode.experimental_string_processing:
            if line.inside_brackets:
                transformers = [string_merge, string_paren_strip, string_split, delimiter_split, standalone_comment_split, string_paren_wrap, rhs]
            else:
                transformers = [string_merge, string_paren_strip, string_split, string_paren_wrap, rhs]
        elif line.inside_brackets:
            transformers = [delimiter_split, standalone_comment_split, rhs]
        else:
            transformers = [rhs]
    for transform in transformers:
        try:
            result = run_transformer(line, transform, mode, features, line_str=line_str)
        except CannotTransform:
            continue
        else:
            (yield from result)
            break
    else:
        (yield line)

@dataclass
class StringTransformer(ABC):
    '\n    An implementation of the Transformer protocol that relies on its\n    subclasses overriding the template methods `do_match(...)` and\n    `do_transform(...)`.\n\n    This Transformer works exclusively on strings (for example, by merging\n    or splitting them).\n\n    The following sections can be found among the docstrings of each concrete\n    StringTransformer subclass.\n\n    Requirements:\n        Which requirements must be met of the given Line for this\n        StringTransformer to be applied?\n\n    Transformations:\n        If the given Line meets all of the above requirements, which string\n        transformations can you expect to be applied to it by this\n        StringTransformer?\n\n    Collaborations:\n        What contractual agreements does this StringTransformer have with other\n        StringTransfomers? Such collaborations should be eliminated/minimized\n        as much as possible.\n    '
    __name__ = 'StringTransformer'

    @abstractmethod
    def do_match(self, line):
        '\n        Returns:\n            * Ok(string_idx) such that `line.leaves[string_idx]` is our target\n            string, if a match was able to be made.\n                OR\n            * Err(CannotTransform), if a match was not able to be made.\n        '

    @abstractmethod
    def do_transform(self, line, string_idx):
        "\n        Yields:\n            * Ok(new_line) where new_line is the new transformed line.\n                OR\n            * Err(CannotTransform) if the transformation failed for some reason. The\n            `do_match(...)` template method should usually be used to reject\n            the form of the given Line, but in some cases it is difficult to\n            know whether or not a Line meets the StringTransformer's\n            requirements until the transformation is already midway.\n\n        Side Effects:\n            This method should NOT mutate @line directly, but it MAY mutate the\n            Line's underlying Node structure. (WARNING: If the underlying Node\n            structure IS altered, then this method should NOT be allowed to\n            yield an CannotTransform after that point.)\n        "

    def __call__(self, line, _features):
        '\n        StringTransformer instances have a call signature that mirrors that of\n        the Transformer type.\n\n        Raises:\n            CannotTransform(...) if the concrete StringTransformer class is unable\n            to transform @line.\n        '
        if (not any(((leaf.type == token.STRING) for leaf in line.leaves))):
            raise CannotTransform('There are no strings in this line.')
        match_result = self.do_match(line)
        if isinstance(match_result, Err):
            cant_transform = match_result.err()
            raise CannotTransform(f'The string transformer {self.__class__.__name__} does not recognize this line as one that it can transform.') from cant_transform
        string_idx = match_result.ok()
        for line_result in self.do_transform(line, string_idx):
            if isinstance(line_result, Err):
                cant_transform = line_result.err()
                raise CannotTransform('StringTransformer failed while attempting to transform string.') from cant_transform
            line = line_result.ok()
            (yield line)

@dataclass
class CustomSplit():
    'A custom (i.e. manual) string split.\n\n    A single CustomSplit instance represents a single substring.\n\n    Examples:\n        Consider the following string:\n        ```\n        "Hi there friend."\n        " This is a custom"\n        f" string {split}."\n        ```\n\n        This string will correspond to the following three CustomSplit instances:\n        ```\n        CustomSplit(False, 16)\n        CustomSplit(False, 17)\n        CustomSplit(True, 16)\n        ```\n    '

class CustomSplitMapMixin():
    '\n    This mixin class is used to map merged strings to a sequence of\n    CustomSplits, which will then be used to re-split the strings iff none of\n    the resultant substrings go over the configured max line length.\n    '
    _Key = Tuple[(StringID, str)]
    _CUSTOM_SPLIT_MAP = defaultdict(tuple)

    @staticmethod
    def _get_key(string):
        '\n        Returns:\n            A unique identifier that is used internally to map @string to a\n            group of custom splits.\n        '
        return (id(string), string)

    def add_custom_splits(self, string, custom_splits):
        'Custom Split Map Setter Method\n\n        Side Effects:\n            Adds a mapping from @string to the custom splits @custom_splits.\n        '
        key = self._get_key(string)
        self._CUSTOM_SPLIT_MAP[key] = tuple(custom_splits)

    def pop_custom_splits(self, string):
        'Custom Split Map Getter Method\n\n        Returns:\n            * A list of the custom splits that are mapped to @string, if any\n            exist.\n                OR\n            * [], otherwise.\n\n        Side Effects:\n            Deletes the mapping between @string and its associated custom\n            splits (which are returned to the caller).\n        '
        key = self._get_key(string)
        custom_splits = self._CUSTOM_SPLIT_MAP[key]
        del self._CUSTOM_SPLIT_MAP[key]
        return list(custom_splits)

    def has_custom_splits(self, string):
        '\n        Returns:\n            True iff @string is associated with a set of custom splits.\n        '
        key = self._get_key(string)
        return (key in self._CUSTOM_SPLIT_MAP)

class StringMerger(CustomSplitMapMixin, StringTransformer):
    "StringTransformer that merges strings together.\n\n    Requirements:\n        (A) The line contains adjacent strings such that ALL of the validation checks\n        listed in StringMerger.__validate_msg(...)'s docstring pass.\n            OR\n        (B) The line contains a string which uses line continuation backslashes.\n\n    Transformations:\n        Depending on which of the two requirements above where met, either:\n\n        (A) The string group associated with the target string is merged.\n            OR\n        (B) All line-continuation backslashes are removed from the target string.\n\n    Collaborations:\n        StringMerger provides custom split information to StringSplitter.\n    "

    def do_match(self, line):
        LL = line.leaves
        is_valid_index = is_valid_index_factory(LL)
        for (i, leaf) in enumerate(LL):
            if ((leaf.type == token.STRING) and is_valid_index((i + 1)) and (LL[(i + 1)].type == token.STRING)):
                return Ok(i)
            if ((leaf.type == token.STRING) and ('\\\n' in leaf.value)):
                return Ok(i)
        return TErr('This line has no strings that need merging.')

    def do_transform(self, line, string_idx):
        new_line = line
        rblc_result = self.__remove_backslash_line_continuation_chars(new_line, string_idx)
        if isinstance(rblc_result, Ok):
            new_line = rblc_result.ok()
        msg_result = self.__merge_string_group(new_line, string_idx)
        if isinstance(msg_result, Ok):
            new_line = msg_result.ok()
        if (isinstance(rblc_result, Err) and isinstance(msg_result, Err)):
            msg_cant_transform = msg_result.err()
            rblc_cant_transform = rblc_result.err()
            cant_transform = CannotTransform('StringMerger failed to merge any strings in this line.')
            msg_cant_transform.__cause__ = rblc_cant_transform
            cant_transform.__cause__ = msg_cant_transform
            (yield Err(cant_transform))
        else:
            (yield Ok(new_line))

    @staticmethod
    def __remove_backslash_line_continuation_chars(line, string_idx):
        '\n        Merge strings that were split across multiple lines using\n        line-continuation backslashes.\n\n        Returns:\n            Ok(new_line), if @line contains backslash line-continuation\n            characters.\n                OR\n            Err(CannotTransform), otherwise.\n        '
        LL = line.leaves
        string_leaf = LL[string_idx]
        if (not ((string_leaf.type == token.STRING) and ('\\\n' in string_leaf.value) and (not has_triple_quotes(string_leaf.value)))):
            return TErr(f'String leaf {string_leaf} does not contain any backslash line continuation characters.')
        new_line = line.clone()
        new_line.comments = line.comments.copy()
        append_leaves(new_line, line, LL)
        new_string_leaf = new_line.leaves[string_idx]
        new_string_leaf.value = new_string_leaf.value.replace('\\\n', '')
        return Ok(new_line)

    def __merge_string_group(self, line, string_idx):
        '\n        Merges string group (i.e. set of adjacent strings) where the first\n        string in the group is `line.leaves[string_idx]`.\n\n        Returns:\n            Ok(new_line), if ALL of the validation checks found in\n            __validate_msg(...) pass.\n                OR\n            Err(CannotTransform), otherwise.\n        '
        LL = line.leaves
        is_valid_index = is_valid_index_factory(LL)
        vresult = self.__validate_msg(line, string_idx)
        if isinstance(vresult, Err):
            return vresult
        atom_node = LL[string_idx].parent
        BREAK_MARK = '@@@@@ BLACK BREAKPOINT MARKER @@@@@'
        QUOTE = LL[string_idx].value[(- 1)]

        def make_naked(string: str, string_prefix: str) -> str:
            'Strip @string (i.e. make it a "naked" string)\n\n            Pre-conditions:\n                * assert_is_leaf_string(@string)\n\n            Returns:\n                A string that is identical to @string except that\n                @string_prefix has been stripped, the surrounding QUOTE\n                characters have been removed, and any remaining QUOTE\n                characters have been escaped.\n            '
            assert_is_leaf_string(string)
            RE_EVEN_BACKSLASHES = '(?:(?<!\\\\)(?:\\\\\\\\)*)'
            naked_string = string[(len(string_prefix) + 1):(- 1)]
            naked_string = re.sub(((('(' + RE_EVEN_BACKSLASHES) + ')') + QUOTE), ('\\1\\\\' + QUOTE), naked_string)
            return naked_string
        custom_splits = []
        prefix_tracker = []
        next_str_idx = string_idx
        prefix = ''
        while ((not prefix) and is_valid_index(next_str_idx) and (LL[next_str_idx].type == token.STRING)):
            prefix = get_string_prefix(LL[next_str_idx].value)
            next_str_idx += 1
        S = ''
        NS = ''
        num_of_strings = 0
        next_str_idx = string_idx
        while (is_valid_index(next_str_idx) and (LL[next_str_idx].type == token.STRING)):
            num_of_strings += 1
            SS = LL[next_str_idx].value
            next_prefix = get_string_prefix(SS)
            if (('f' in prefix) and ('f' not in next_prefix)):
                SS = re.subf('(\\{|\\})', '{1}{1}', SS)
            NSS = make_naked(SS, next_prefix)
            has_prefix = bool(next_prefix)
            prefix_tracker.append(has_prefix)
            S = (((((prefix + QUOTE) + NS) + NSS) + BREAK_MARK) + QUOTE)
            NS = make_naked(S, prefix)
            next_str_idx += 1
        S_leaf = Leaf(token.STRING, S)
        if self.normalize_strings:
            normalize_string_quotes(S_leaf)
        temp_string = S_leaf.value[(len(prefix) + 1):(- 1)]
        for has_prefix in prefix_tracker:
            mark_idx = temp_string.find(BREAK_MARK)
            assert (mark_idx >= 0), 'Logic error while filling the custom string breakpoint cache.'
            temp_string = temp_string[(mark_idx + len(BREAK_MARK)):]
            breakpoint_idx = ((mark_idx + (len(prefix) if has_prefix else 0)) + 1)
            custom_splits.append(CustomSplit(has_prefix, breakpoint_idx))
        string_leaf = Leaf(token.STRING, S_leaf.value.replace(BREAK_MARK, ''))
        if (atom_node is not None):
            replace_child(atom_node, string_leaf)
        new_line = line.clone()
        for (i, leaf) in enumerate(LL):
            if (i == string_idx):
                new_line.append(string_leaf)
            if (string_idx <= i < (string_idx + num_of_strings)):
                for comment_leaf in line.comments_after(LL[i]):
                    new_line.append(comment_leaf, preformatted=True)
                continue
            append_leaves(new_line, line, [leaf])
        self.add_custom_splits(string_leaf.value, custom_splits)
        return Ok(new_line)

    @staticmethod
    def __validate_msg(line, string_idx):
        'Validate (M)erge (S)tring (G)roup\n\n        Transform-time string validation logic for __merge_string_group(...).\n\n        Returns:\n            * Ok(None), if ALL validation checks (listed below) pass.\n                OR\n            * Err(CannotTransform), if any of the following are true:\n                - The target string group does not contain ANY stand-alone comments.\n                - The target string is not in a string group (i.e. it has no\n                  adjacent strings).\n                - The string group has more than one inline comment.\n                - The string group has an inline comment that appears to be a pragma.\n                - The set of all string prefixes in the string group is of\n                  length greater than one and is not equal to {"", "f"}.\n                - The string group consists of raw strings.\n        '
        for inc in [1, (- 1)]:
            i = string_idx
            found_sa_comment = False
            is_valid_index = is_valid_index_factory(line.leaves)
            while (is_valid_index(i) and (line.leaves[i].type in [token.STRING, STANDALONE_COMMENT])):
                if (line.leaves[i].type == STANDALONE_COMMENT):
                    found_sa_comment = True
                elif found_sa_comment:
                    return TErr('StringMerger does NOT merge string groups which contain stand-alone comments.')
                i += inc
        num_of_inline_string_comments = 0
        set_of_prefixes = set()
        num_of_strings = 0
        for leaf in line.leaves[string_idx:]:
            if (leaf.type != token.STRING):
                if ((leaf.type == token.COMMA) and (id(leaf) in line.comments)):
                    num_of_inline_string_comments += 1
                break
            if has_triple_quotes(leaf.value):
                return TErr('StringMerger does NOT merge multiline strings.')
            num_of_strings += 1
            prefix = get_string_prefix(leaf.value)
            if ('r' in prefix):
                return TErr('StringMerger does NOT merge raw strings.')
            set_of_prefixes.add(prefix)
            if (id(leaf) in line.comments):
                num_of_inline_string_comments += 1
                if contains_pragma_comment(line.comments[id(leaf)]):
                    return TErr('Cannot merge strings which have pragma comments.')
        if (num_of_strings < 2):
            return TErr(f'Not enough strings to merge (num_of_strings={num_of_strings}).')
        if (num_of_inline_string_comments > 1):
            return TErr(f'Too many inline string comments ({num_of_inline_string_comments}).')
        if ((len(set_of_prefixes) > 1) and (set_of_prefixes != {'', 'f'})):
            return TErr(f'Too many different prefixes ({set_of_prefixes}).')
        return Ok(None)

class StringParenStripper(StringTransformer):
    'StringTransformer that strips surrounding parentheses from strings.\n\n    Requirements:\n        The line contains a string which is surrounded by parentheses and:\n            - The target string is NOT the only argument to a function call.\n            - The target string is NOT a "pointless" string.\n            - If the target string contains a PERCENT, the brackets are not\n              preceeded or followed by an operator with higher precedence than\n              PERCENT.\n\n    Transformations:\n        The parentheses mentioned in the \'Requirements\' section are stripped.\n\n    Collaborations:\n        StringParenStripper has its own inherent usefulness, but it is also\n        relied on to clean up the parentheses created by StringParenWrapper (in\n        the event that they are no longer needed).\n    '

    def do_match(self, line):
        LL = line.leaves
        is_valid_index = is_valid_index_factory(LL)
        for (idx, leaf) in enumerate(LL):
            if (leaf.type != token.STRING):
                continue
            if (leaf.parent and leaf.parent.parent and (leaf.parent.parent.type == syms.simple_stmt)):
                continue
            if ((not is_valid_index((idx - 1))) or (LL[(idx - 1)].type != token.LPAR) or is_empty_lpar(LL[(idx - 1)])):
                continue
            if (is_valid_index((idx - 2)) and ((LL[(idx - 2)].type == token.NAME) or (LL[(idx - 2)].type in CLOSING_BRACKETS))):
                continue
            string_idx = idx
            string_parser = StringParser()
            next_idx = string_parser.parse(LL, string_idx)
            if is_valid_index((idx - 2)):
                before_lpar = LL[(idx - 2)]
                if ((token.PERCENT in {leaf.type for leaf in LL[(idx - 1):next_idx]}) and ((before_lpar.type in {token.STAR, token.AT, token.SLASH, token.DOUBLESLASH, token.PERCENT, token.TILDE, token.DOUBLESTAR, token.AWAIT, token.LSQB, token.LPAR}) or (before_lpar.parent and (before_lpar.parent.type == syms.factor) and (before_lpar.type in {token.PLUS, token.MINUS})))):
                    continue
            if (is_valid_index(next_idx) and (LL[next_idx].type == token.RPAR) and (not is_empty_rpar(LL[next_idx]))):
                if (is_valid_index((next_idx + 1)) and (LL[(next_idx + 1)].type in {token.DOUBLESTAR, token.LSQB, token.LPAR, token.DOT})):
                    continue
                return Ok(string_idx)
        return TErr('This line has no strings wrapped in parens.')

    def do_transform(self, line, string_idx):
        LL = line.leaves
        string_parser = StringParser()
        rpar_idx = string_parser.parse(LL, string_idx)
        for leaf in (LL[(string_idx - 1)], LL[rpar_idx]):
            if line.comments_after(leaf):
                (yield TErr('Will not strip parentheses which have comments attached to them.'))
                return
        new_line = line.clone()
        new_line.comments = line.comments.copy()
        try:
            append_leaves(new_line, line, LL[:(string_idx - 1)])
        except BracketMatchError:
            append_leaves(new_line, line, LL[:(string_idx - 1)], preformatted=True)
        string_leaf = Leaf(token.STRING, LL[string_idx].value)
        LL[(string_idx - 1)].remove()
        replace_child(LL[string_idx], string_leaf)
        new_line.append(string_leaf)
        append_leaves(new_line, line, (LL[(string_idx + 1):rpar_idx] + LL[(rpar_idx + 1):]))
        LL[rpar_idx].remove()
        (yield Ok(new_line))

class BaseStringSplitter(StringTransformer):
    '\n    Abstract class for StringTransformers which transform a Line\'s strings by splitting\n    them or placing them on their own lines where necessary to avoid going over\n    the configured line length.\n\n    Requirements:\n        * The target string value is responsible for the line going over the\n        line length limit. It follows that after all of black\'s other line\n        split methods have been exhausted, this line (or one of the resulting\n        lines after all line splits are performed) would still be over the\n        line_length limit unless we split this string.\n            AND\n        * The target string is NOT a "pointless" string (i.e. a string that has\n        no parent or siblings).\n            AND\n        * The target string is not followed by an inline comment that appears\n        to be a pragma.\n            AND\n        * The target string is not a multiline (i.e. triple-quote) string.\n    '

    @abstractmethod
    def do_splitter_match(self, line):
        '\n        BaseStringSplitter asks its clients to override this method instead of\n        `StringTransformer.do_match(...)`.\n\n        Follows the same protocol as `StringTransformer.do_match(...)`.\n\n        Refer to `help(StringTransformer.do_match)` for more information.\n        '

    def do_match(self, line):
        match_result = self.do_splitter_match(line)
        if isinstance(match_result, Err):
            return match_result
        string_idx = match_result.ok()
        vresult = self.__validate(line, string_idx)
        if isinstance(vresult, Err):
            return vresult
        return match_result

    def __validate(self, line, string_idx):
        "\n        Checks that @line meets all of the requirements listed in this classes'\n        docstring. Refer to `help(BaseStringSplitter)` for a detailed\n        description of those requirements.\n\n        Returns:\n            * Ok(None), if ALL of the requirements are met.\n                OR\n            * Err(CannotTransform), if ANY of the requirements are NOT met.\n        "
        LL = line.leaves
        string_leaf = LL[string_idx]
        max_string_length = self.__get_max_string_length(line, string_idx)
        if (len(string_leaf.value) <= max_string_length):
            return TErr('The string itself is not what is causing this line to be too long.')
        if ((not string_leaf.parent) or ([L.type for L in string_leaf.parent.children] == [token.STRING, token.NEWLINE])):
            return TErr(f'This string ({string_leaf.value}) appears to be pointless (i.e. has no parent).')
        if ((id(line.leaves[string_idx]) in line.comments) and contains_pragma_comment(line.comments[id(line.leaves[string_idx])])):
            return TErr("Line appears to end with an inline pragma comment. Splitting the line could modify the pragma's behavior.")
        if has_triple_quotes(string_leaf.value):
            return TErr('We cannot split multiline strings.')
        return Ok(None)

    def __get_max_string_length(self, line, string_idx):
        '\n        Calculates the max string length used when attempting to determine\n        whether or not the target string is responsible for causing the line to\n        go over the line length limit.\n\n        WARNING: This method is tightly coupled to both StringSplitter and\n        (especially) StringParenWrapper. There is probably a better way to\n        accomplish what is being done here.\n\n        Returns:\n            max_string_length: such that `line.leaves[string_idx].value >\n            max_string_length` implies that the target string IS responsible\n            for causing this line to exceed the line length limit.\n        '
        LL = line.leaves
        is_valid_index = is_valid_index_factory(LL)
        offset = (line.depth * 4)
        if is_valid_index((string_idx - 1)):
            p_idx = (string_idx - 1)
            if ((LL[(string_idx - 1)].type == token.LPAR) and (LL[(string_idx - 1)].value == '') and (string_idx >= 2)):
                p_idx -= 1
            P = LL[p_idx]
            if (P.type == token.PLUS):
                offset += 2
            if (P.type == token.COMMA):
                offset += 3
            if (P.type in [token.COLON, token.EQUAL, token.NAME]):
                offset += 1
                for leaf in reversed(LL[:(p_idx + 1)]):
                    offset += len(str(leaf))
                    if (leaf.type in CLOSING_BRACKETS):
                        break
        if is_valid_index((string_idx + 1)):
            N = LL[(string_idx + 1)]
            if ((N.type == token.RPAR) and (N.value == '') and (len(LL) > (string_idx + 2))):
                N = LL[(string_idx + 2)]
            if (N.type == token.COMMA):
                offset += 1
            if is_valid_index((string_idx + 2)):
                NN = LL[(string_idx + 2)]
                if ((N.type == token.DOT) and (NN.type == token.NAME)):
                    offset += 1
                    if (is_valid_index((string_idx + 3)) and (LL[(string_idx + 3)].type == token.LPAR)):
                        offset += 1
                    offset += len(NN.value)
        has_comments = False
        for comment_leaf in line.comments_after(LL[string_idx]):
            if (not has_comments):
                has_comments = True
                offset += 2
            offset += len(comment_leaf.value)
        max_string_length = (self.line_length - offset)
        return max_string_length

class StringSplitter(CustomSplitMapMixin, BaseStringSplitter):
    '\n    StringTransformer that splits "atom" strings (i.e. strings which exist on\n    lines by themselves).\n\n    Requirements:\n        * The line consists ONLY of a single string (with the exception of a\n        \'+\' symbol which MAY exist at the start of the line), MAYBE a string\n        trailer, and MAYBE a trailing comma.\n            AND\n        * All of the requirements listed in BaseStringSplitter\'s docstring.\n\n    Transformations:\n        The string mentioned in the \'Requirements\' section is split into as\n        many substrings as necessary to adhere to the configured line length.\n\n        In the final set of substrings, no substring should be smaller than\n        MIN_SUBSTR_SIZE characters.\n\n        The string will ONLY be split on spaces (i.e. each new substring should\n        start with a space). Note that the string will NOT be split on a space\n        which is escaped with a backslash.\n\n        If the string is an f-string, it will NOT be split in the middle of an\n        f-expression (e.g. in f"FooBar: {foo() if x else bar()}", {foo() if x\n        else bar()} is an f-expression).\n\n        If the string that is being split has an associated set of custom split\n        records and those custom splits will NOT result in any line going over\n        the configured line length, those custom splits are used. Otherwise the\n        string is split as late as possible (from left-to-right) while still\n        adhering to the transformation rules listed above.\n\n    Collaborations:\n        StringSplitter relies on StringMerger to construct the appropriate\n        CustomSplit objects and add them to the custom split map.\n    '
    MIN_SUBSTR_SIZE = 6
    RE_FEXPR = '\n    (?<!\\{) (?:\\{\\{)* \\{ (?!\\{)\n        (?:\n            [^\\{\\}]\n            | \\{\\{\n            | \\}\\}\n            | (?R)\n        )+?\n    (?<!\\}) \\} (?:\\}\\})* (?!\\})\n    '

    def do_splitter_match(self, line):
        LL = line.leaves
        is_valid_index = is_valid_index_factory(LL)
        idx = 0
        if (is_valid_index(idx) and (LL[idx].type == token.PLUS)):
            idx += 1
        if (is_valid_index(idx) and is_empty_lpar(LL[idx])):
            idx += 1
        if ((not is_valid_index(idx)) or (LL[idx].type != token.STRING)):
            return TErr('Line does not start with a string.')
        string_idx = idx
        string_parser = StringParser()
        idx = string_parser.parse(LL, string_idx)
        if (is_valid_index(idx) and is_empty_rpar(LL[idx])):
            idx += 1
        if (is_valid_index(idx) and (LL[idx].type == token.COMMA)):
            idx += 1
        if is_valid_index(idx):
            return TErr('This line does not end with a string.')
        return Ok(string_idx)

    def do_transform(self, line, string_idx):
        LL = line.leaves
        QUOTE = LL[string_idx].value[(- 1)]
        is_valid_index = is_valid_index_factory(LL)
        insert_str_child = insert_str_child_factory(LL[string_idx])
        prefix = get_string_prefix(LL[string_idx].value)
        drop_pointless_f_prefix = (('f' in prefix) and re.search(self.RE_FEXPR, LL[string_idx].value, re.VERBOSE))
        first_string_line = True
        starts_with_plus = (LL[0].type == token.PLUS)

        def line_needs_plus() -> bool:
            return (first_string_line and starts_with_plus)

        def maybe_append_plus(new_line: Line) -> None:
            '\n            Side Effects:\n                If @line starts with a plus and this is the first line we are\n                constructing, this function appends a PLUS leaf to @new_line\n                and replaces the old PLUS leaf in the node structure. Otherwise\n                this function does nothing.\n            '
            if line_needs_plus():
                plus_leaf = Leaf(token.PLUS, '+')
                replace_child(LL[0], plus_leaf)
                new_line.append(plus_leaf)
        ends_with_comma = (is_valid_index((string_idx + 1)) and (LL[(string_idx + 1)].type == token.COMMA))

        def max_last_string() -> int:
            '\n            Returns:\n                The max allowed length of the string value used for the last\n                line we will construct.\n            '
            result = self.line_length
            result -= (line.depth * 4)
            result -= (1 if ends_with_comma else 0)
            result -= (2 if line_needs_plus() else 0)
            return result
        max_break_idx = self.line_length
        max_break_idx -= 1
        max_break_idx -= (line.depth * 4)
        if (max_break_idx < 0):
            (yield TErr(f'Unable to split {LL[string_idx].value} at such high of a line depth: {line.depth}'))
            return
        custom_splits = self.pop_custom_splits(LL[string_idx].value)
        use_custom_breakpoints = bool((custom_splits and all(((csplit.break_idx <= max_break_idx) for csplit in custom_splits))))
        rest_value = LL[string_idx].value

        def more_splits_should_be_made() -> bool:
            '\n            Returns:\n                True iff `rest_value` (the remaining string value from the last\n                split), should be split again.\n            '
            if use_custom_breakpoints:
                return (len(custom_splits) > 1)
            else:
                return (len(rest_value) > max_last_string())
        string_line_results: List[Ok[Line]] = []
        while more_splits_should_be_made():
            if use_custom_breakpoints:
                csplit = custom_splits.pop(0)
                break_idx = csplit.break_idx
            else:
                max_bidx = ((max_break_idx - 2) if line_needs_plus() else max_break_idx)
                maybe_break_idx = self.__get_break_idx(rest_value, max_bidx)
                if (maybe_break_idx is None):
                    if custom_splits:
                        rest_value = LL[string_idx].value
                        string_line_results = []
                        first_string_line = True
                        use_custom_breakpoints = True
                        continue
                    break
                break_idx = maybe_break_idx
            next_value = (rest_value[:break_idx] + QUOTE)
            if (drop_pointless_f_prefix and (next_value != self.__normalize_f_string(next_value, prefix))):
                break_idx = ((break_idx + 1) if (use_custom_breakpoints and (not csplit.has_prefix)) else break_idx)
                next_value = (rest_value[:break_idx] + QUOTE)
                next_value = self.__normalize_f_string(next_value, prefix)
            next_leaf = Leaf(token.STRING, next_value)
            insert_str_child(next_leaf)
            self.__maybe_normalize_string_quotes(next_leaf)
            next_line = line.clone()
            maybe_append_plus(next_line)
            next_line.append(next_leaf)
            string_line_results.append(Ok(next_line))
            rest_value = ((prefix + QUOTE) + rest_value[break_idx:])
            first_string_line = False
        (yield from string_line_results)
        if drop_pointless_f_prefix:
            rest_value = self.__normalize_f_string(rest_value, prefix)
        rest_leaf = Leaf(token.STRING, rest_value)
        insert_str_child(rest_leaf)
        self.__maybe_normalize_string_quotes(rest_leaf)
        last_line = line.clone()
        maybe_append_plus(last_line)
        if is_valid_index((string_idx + 1)):
            temp_value = rest_value
            for leaf in LL[(string_idx + 1):]:
                temp_value += str(leaf)
                if (leaf.type == token.LPAR):
                    break
            if ((len(temp_value) <= max_last_string()) or (LL[(string_idx + 1)].type == token.COMMA)):
                last_line.append(rest_leaf)
                append_leaves(last_line, line, LL[(string_idx + 1):])
                (yield Ok(last_line))
            else:
                last_line.append(rest_leaf)
                (yield Ok(last_line))
                non_string_line = line.clone()
                append_leaves(non_string_line, line, LL[(string_idx + 1):])
                (yield Ok(non_string_line))
        else:
            last_line.append(rest_leaf)
            last_line.comments = line.comments.copy()
            (yield Ok(last_line))

    def __get_break_idx(self, string, max_break_idx):
        "\n        This method contains the algorithm that StringSplitter uses to\n        determine which character to split each string at.\n\n        Args:\n            @string: The substring that we are attempting to split.\n            @max_break_idx: The ideal break index. We will return this value if it\n            meets all the necessary conditions. In the likely event that it\n            doesn't we will try to find the closest index BELOW @max_break_idx\n            that does. If that fails, we will expand our search by also\n            considering all valid indices ABOVE @max_break_idx.\n\n        Pre-Conditions:\n            * assert_is_leaf_string(@string)\n            * 0 <= @max_break_idx < len(@string)\n\n        Returns:\n            break_idx, if an index is able to be found that meets all of the\n            conditions listed in the 'Transformations' section of this classes'\n            docstring.\n                OR\n            None, otherwise.\n        "
        is_valid_index = is_valid_index_factory(string)
        assert is_valid_index(max_break_idx)
        assert_is_leaf_string(string)
        _fexpr_slices: Optional[List[Tuple[(Index, Index)]]] = None

        def fexpr_slices() -> Iterator[Tuple[(Index, Index)]]:
            '\n            Yields:\n                All ranges of @string which, if @string were to be split there,\n                would result in the splitting of an f-expression (which is NOT\n                allowed).\n            '
            nonlocal _fexpr_slices
            if (_fexpr_slices is None):
                _fexpr_slices = []
                for match in re.finditer(self.RE_FEXPR, string, re.VERBOSE):
                    _fexpr_slices.append(match.span())
            (yield from _fexpr_slices)
        is_fstring = ('f' in get_string_prefix(string))

        def breaks_fstring_expression(i: Index) -> bool:
            '\n            Returns:\n                True iff returning @i would result in the splitting of an\n                f-expression (which is NOT allowed).\n            '
            if (not is_fstring):
                return False
            for (start, end) in fexpr_slices():
                if (start <= i < end):
                    return True
            return False

        def passes_all_checks(i: Index) -> bool:
            "\n            Returns:\n                True iff ALL of the conditions listed in the 'Transformations'\n                section of this classes' docstring would be be met by returning @i.\n            "
            is_space = (string[i] == ' ')
            is_not_escaped = True
            j = (i - 1)
            while (is_valid_index(j) and (string[j] == '\\')):
                is_not_escaped = (not is_not_escaped)
                j -= 1
            is_big_enough = ((len(string[i:]) >= self.MIN_SUBSTR_SIZE) and (len(string[:i]) >= self.MIN_SUBSTR_SIZE))
            return (is_space and is_not_escaped and is_big_enough and (not breaks_fstring_expression(i)))
        break_idx = max_break_idx
        while (is_valid_index((break_idx - 1)) and (not passes_all_checks(break_idx))):
            break_idx -= 1
        if (not passes_all_checks(break_idx)):
            break_idx = (max_break_idx + 1)
            while (is_valid_index((break_idx + 1)) and (not passes_all_checks(break_idx))):
                break_idx += 1
            if ((not is_valid_index(break_idx)) or (not passes_all_checks(break_idx))):
                return None
        return break_idx

    def __maybe_normalize_string_quotes(self, leaf):
        if self.normalize_strings:
            normalize_string_quotes(leaf)

    def __normalize_f_string(self, string, prefix):
        "\n        Pre-Conditions:\n            * assert_is_leaf_string(@string)\n\n        Returns:\n            * If @string is an f-string that contains no f-expressions, we\n            return a string identical to @string except that the 'f' prefix\n            has been stripped and all double braces (i.e. '{{' or '}}') have\n            been normalized (i.e. turned into '{' or '}').\n                OR\n            * Otherwise, we return @string.\n        "
        assert_is_leaf_string(string)
        if (('f' in prefix) and (not re.search(self.RE_FEXPR, string, re.VERBOSE))):
            new_prefix = prefix.replace('f', '')
            temp = string[len(prefix):]
            temp = re.sub('\\{\\{', '{', temp)
            temp = re.sub('\\}\\}', '}', temp)
            new_string = temp
            return f'{new_prefix}{new_string}'
        else:
            return string

class StringParenWrapper(CustomSplitMapMixin, BaseStringSplitter):
    '\n    StringTransformer that splits non-"atom" strings (i.e. strings that do not\n    exist on lines by themselves).\n\n    Requirements:\n        All of the requirements listed in BaseStringSplitter\'s docstring in\n        addition to the requirements listed below:\n\n        * The line is a return/yield statement, which returns/yields a string.\n            OR\n        * The line is part of a ternary expression (e.g. `x = y if cond else\n        z`) such that the line starts with `else <string>`, where <string> is\n        some string.\n            OR\n        * The line is an assert statement, which ends with a string.\n            OR\n        * The line is an assignment statement (e.g. `x = <string>` or `x +=\n        <string>`) such that the variable is being assigned the value of some\n        string.\n            OR\n        * The line is a dictionary key assignment where some valid key is being\n        assigned the value of some string.\n\n    Transformations:\n        The chosen string is wrapped in parentheses and then split at the LPAR.\n\n        We then have one line which ends with an LPAR and another line that\n        starts with the chosen string. The latter line is then split again at\n        the RPAR. This results in the RPAR (and possibly a trailing comma)\n        being placed on its own line.\n\n        NOTE: If any leaves exist to the right of the chosen string (except\n        for a trailing comma, which would be placed after the RPAR), those\n        leaves are placed inside the parentheses.  In effect, the chosen\n        string is not necessarily being "wrapped" by parentheses. We can,\n        however, count on the LPAR being placed directly before the chosen\n        string.\n\n        In other words, StringParenWrapper creates "atom" strings. These\n        can then be split again by StringSplitter, if necessary.\n\n    Collaborations:\n        In the event that a string line split by StringParenWrapper is\n        changed such that it no longer needs to be given its own line,\n        StringParenWrapper relies on StringParenStripper to clean up the\n        parentheses it created.\n    '

    def do_splitter_match(self, line):
        LL = line.leaves
        string_idx = (self._return_match(LL) or self._else_match(LL) or self._assert_match(LL) or self._assign_match(LL) or self._dict_match(LL))
        if (string_idx is not None):
            string_value = line.leaves[string_idx].value
            if (' ' not in string_value):
                max_string_length = (self.line_length - ((line.depth + 1) * 4))
                if (len(string_value) > max_string_length):
                    if (not self.has_custom_splits(string_value)):
                        return TErr("We do not wrap long strings in parentheses when the resultant line would still be over the specified line length and can't be split further by StringSplitter.")
            return Ok(string_idx)
        return TErr('This line does not contain any non-atomic strings.')

    @staticmethod
    def _return_match(LL):
        "\n        Returns:\n            string_idx such that @LL[string_idx] is equal to our target (i.e.\n            matched) string, if this line matches the return/yield statement\n            requirements listed in the 'Requirements' section of this classes'\n            docstring.\n                OR\n            None, otherwise.\n        "
        if ((parent_type(LL[0]) in [syms.return_stmt, syms.yield_expr]) and (LL[0].value in ['return', 'yield'])):
            is_valid_index = is_valid_index_factory(LL)
            idx = (2 if (is_valid_index(1) and is_empty_par(LL[1])) else 1)
            if (is_valid_index(idx) and (LL[idx].type == token.STRING)):
                return idx
        return None

    @staticmethod
    def _else_match(LL):
        "\n        Returns:\n            string_idx such that @LL[string_idx] is equal to our target (i.e.\n            matched) string, if this line matches the ternary expression\n            requirements listed in the 'Requirements' section of this classes'\n            docstring.\n                OR\n            None, otherwise.\n        "
        if ((parent_type(LL[0]) == syms.test) and (LL[0].type == token.NAME) and (LL[0].value == 'else')):
            is_valid_index = is_valid_index_factory(LL)
            idx = (2 if (is_valid_index(1) and is_empty_par(LL[1])) else 1)
            if (is_valid_index(idx) and (LL[idx].type == token.STRING)):
                return idx
        return None

    @staticmethod
    def _assert_match(LL):
        "\n        Returns:\n            string_idx such that @LL[string_idx] is equal to our target (i.e.\n            matched) string, if this line matches the assert statement\n            requirements listed in the 'Requirements' section of this classes'\n            docstring.\n                OR\n            None, otherwise.\n        "
        if ((parent_type(LL[0]) == syms.assert_stmt) and (LL[0].value == 'assert')):
            is_valid_index = is_valid_index_factory(LL)
            for (i, leaf) in enumerate(LL):
                if (leaf.type == token.COMMA):
                    idx = ((i + 2) if is_empty_par(LL[(i + 1)]) else (i + 1))
                    if (is_valid_index(idx) and (LL[idx].type == token.STRING)):
                        string_idx = idx
                        string_parser = StringParser()
                        idx = string_parser.parse(LL, string_idx)
                        if (not is_valid_index(idx)):
                            return string_idx
        return None

    @staticmethod
    def _assign_match(LL):
        "\n        Returns:\n            string_idx such that @LL[string_idx] is equal to our target (i.e.\n            matched) string, if this line matches the assignment statement\n            requirements listed in the 'Requirements' section of this classes'\n            docstring.\n                OR\n            None, otherwise.\n        "
        if ((parent_type(LL[0]) in [syms.expr_stmt, syms.argument, syms.power]) and (LL[0].type == token.NAME)):
            is_valid_index = is_valid_index_factory(LL)
            for (i, leaf) in enumerate(LL):
                if (leaf.type in [token.EQUAL, token.PLUSEQUAL]):
                    idx = ((i + 2) if is_empty_par(LL[(i + 1)]) else (i + 1))
                    if (is_valid_index(idx) and (LL[idx].type == token.STRING)):
                        string_idx = idx
                        string_parser = StringParser()
                        idx = string_parser.parse(LL, string_idx)
                        if ((parent_type(LL[0]) == syms.argument) and is_valid_index(idx) and (LL[idx].type == token.COMMA)):
                            idx += 1
                        if (not is_valid_index(idx)):
                            return string_idx
        return None

    @staticmethod
    def _dict_match(LL):
        "\n        Returns:\n            string_idx such that @LL[string_idx] is equal to our target (i.e.\n            matched) string, if this line matches the dictionary key assignment\n            statement requirements listed in the 'Requirements' section of this\n            classes' docstring.\n                OR\n            None, otherwise.\n        "
        if (syms.dictsetmaker in [parent_type(LL[0]), parent_type(LL[0].parent)]):
            is_valid_index = is_valid_index_factory(LL)
            for (i, leaf) in enumerate(LL):
                if (leaf.type == token.COLON):
                    idx = ((i + 2) if is_empty_par(LL[(i + 1)]) else (i + 1))
                    if (is_valid_index(idx) and (LL[idx].type == token.STRING)):
                        string_idx = idx
                        string_parser = StringParser()
                        idx = string_parser.parse(LL, string_idx)
                        if (is_valid_index(idx) and (LL[idx].type == token.COMMA)):
                            idx += 1
                        if (not is_valid_index(idx)):
                            return string_idx
        return None

    def do_transform(self, line, string_idx):
        LL = line.leaves
        is_valid_index = is_valid_index_factory(LL)
        insert_str_child = insert_str_child_factory(LL[string_idx])
        comma_idx = (- 1)
        ends_with_comma = False
        if (LL[comma_idx].type == token.COMMA):
            ends_with_comma = True
        leaves_to_steal_comments_from = [LL[string_idx]]
        if ends_with_comma:
            leaves_to_steal_comments_from.append(LL[comma_idx])
        first_line = line.clone()
        left_leaves = LL[:string_idx]
        old_parens_exist = False
        if (left_leaves and (left_leaves[(- 1)].type == token.LPAR)):
            old_parens_exist = True
            leaves_to_steal_comments_from.append(left_leaves[(- 1)])
            left_leaves.pop()
        append_leaves(first_line, line, left_leaves)
        lpar_leaf = Leaf(token.LPAR, '(')
        if old_parens_exist:
            replace_child(LL[(string_idx - 1)], lpar_leaf)
        else:
            insert_str_child(lpar_leaf)
        first_line.append(lpar_leaf)
        for leaf in leaves_to_steal_comments_from:
            for comment_leaf in line.comments_after(leaf):
                first_line.append(comment_leaf, preformatted=True)
        (yield Ok(first_line))
        string_value = LL[string_idx].value
        string_line = Line(depth=(line.depth + 1), inside_brackets=True, should_explode=line.should_explode)
        string_leaf = Leaf(token.STRING, string_value)
        insert_str_child(string_leaf)
        string_line.append(string_leaf)
        old_rpar_leaf = None
        if is_valid_index((string_idx + 1)):
            right_leaves = LL[(string_idx + 1):]
            if ends_with_comma:
                right_leaves.pop()
            if old_parens_exist:
                assert (right_leaves and (right_leaves[(- 1)].type == token.RPAR)), 'Apparently, old parentheses do NOT exist?!'
                old_rpar_leaf = right_leaves.pop()
            append_leaves(string_line, line, right_leaves)
        (yield Ok(string_line))
        last_line = line.clone()
        last_line.bracket_tracker = first_line.bracket_tracker
        new_rpar_leaf = Leaf(token.RPAR, ')')
        if (old_rpar_leaf is not None):
            replace_child(old_rpar_leaf, new_rpar_leaf)
        else:
            insert_str_child(new_rpar_leaf)
        last_line.append(new_rpar_leaf)
        if ends_with_comma:
            comma_leaf = Leaf(token.COMMA, ',')
            replace_child(LL[comma_idx], comma_leaf)
            last_line.append(comma_leaf)
        (yield Ok(last_line))

class StringParser():
    '\n    A state machine that aids in parsing a string\'s "trailer", which can be\n    either non-existent, an old-style formatting sequence (e.g. `% varX` or `%\n    (varX, varY)`), or a method-call / attribute access (e.g. `.format(varX,\n    varY)`).\n\n    NOTE: A new StringParser object MUST be instantiated for each string\n    trailer we need to parse.\n\n    Examples:\n        We shall assume that `line` equals the `Line` object that corresponds\n        to the following line of python code:\n        ```\n        x = "Some {}.".format("String") + some_other_string\n        ```\n\n        Furthermore, we will assume that `string_idx` is some index such that:\n        ```\n        assert line.leaves[string_idx].value == "Some {}."\n        ```\n\n        The following code snippet then holds:\n        ```\n        string_parser = StringParser()\n        idx = string_parser.parse(line.leaves, string_idx)\n        assert line.leaves[idx].type == token.PLUS\n        ```\n    '
    DEFAULT_TOKEN = (- 1)
    START = 1
    DOT = 2
    NAME = 3
    PERCENT = 4
    SINGLE_FMT_ARG = 5
    LPAR = 6
    RPAR = 7
    DONE = 8
    _goto = {(START, token.DOT): DOT, (START, token.PERCENT): PERCENT, (START, DEFAULT_TOKEN): DONE, (DOT, token.NAME): NAME, (NAME, token.LPAR): LPAR, (NAME, DEFAULT_TOKEN): DONE, (PERCENT, token.LPAR): LPAR, (PERCENT, DEFAULT_TOKEN): SINGLE_FMT_ARG, (SINGLE_FMT_ARG, DEFAULT_TOKEN): DONE, (RPAR, DEFAULT_TOKEN): DONE}

    def __init__(self):
        self._state = self.START
        self._unmatched_lpars = 0

    def parse(self, leaves, string_idx):
        '\n        Pre-conditions:\n            * @leaves[@string_idx].type == token.STRING\n\n        Returns:\n            The index directly after the last leaf which is apart of the string\n            trailer, if a "trailer" exists.\n                OR\n            @string_idx + 1, if no string "trailer" exists.\n        '
        assert (leaves[string_idx].type == token.STRING)
        idx = (string_idx + 1)
        while ((idx < len(leaves)) and self._next_state(leaves[idx])):
            idx += 1
        return idx

    def _next_state(self, leaf):
        "\n        Pre-conditions:\n            * On the first call to this function, @leaf MUST be the leaf that\n            was directly after the string leaf in question (e.g. if our target\n            string is `line.leaves[i]` then the first call to this method must\n            be `line.leaves[i + 1]`).\n            * On the next call to this function, the leaf parameter passed in\n            MUST be the leaf directly following @leaf.\n\n        Returns:\n            True iff @leaf is apart of the string's trailer.\n        "
        if is_empty_par(leaf):
            return True
        next_token = leaf.type
        if (next_token == token.LPAR):
            self._unmatched_lpars += 1
        current_state = self._state
        if (current_state == self.LPAR):
            if (next_token == token.RPAR):
                self._unmatched_lpars -= 1
                if (self._unmatched_lpars == 0):
                    self._state = self.RPAR
        else:
            if ((current_state, next_token) in self._goto):
                self._state = self._goto[(current_state, next_token)]
            elif ((current_state, self.DEFAULT_TOKEN) in self._goto):
                self._state = self._goto[(current_state, self.DEFAULT_TOKEN)]
            else:
                raise RuntimeError(f'{self.__class__.__name__} LOGIC ERROR!')
            if (self._state == self.DONE):
                return False
        return True

def TErr(err_msg):
    '(T)ransform Err\n\n    Convenience function used when working with the TResult type.\n    '
    cant_transform = CannotTransform(err_msg)
    return Err(cant_transform)

def contains_pragma_comment(comment_list):
    '\n    Returns:\n        True iff one of the comments in @comment_list is a pragma used by one\n        of the more common static analysis tools for python (e.g. mypy, flake8,\n        pylint).\n    '
    for comment in comment_list:
        if comment.value.startswith(('# type:', '# noqa', '# pylint:')):
            return True
    return False

def insert_str_child_factory(string_leaf):
    '\n    Factory for a convenience function that is used to orphan @string_leaf\n    and then insert multiple new leaves into the same part of the node\n    structure that @string_leaf had originally occupied.\n\n    Examples:\n        Let `string_leaf = Leaf(token.STRING, \'"foo"\')` and `N =\n        string_leaf.parent`. Assume the node `N` has the following\n        original structure:\n\n        Node(\n            expr_stmt, [\n                Leaf(NAME, \'x\'),\n                Leaf(EQUAL, \'=\'),\n                Leaf(STRING, \'"foo"\'),\n            ]\n        )\n\n        We then run the code snippet shown below.\n        ```\n        insert_str_child = insert_str_child_factory(string_leaf)\n\n        lpar = Leaf(token.LPAR, \'(\')\n        insert_str_child(lpar)\n\n        bar = Leaf(token.STRING, \'"bar"\')\n        insert_str_child(bar)\n\n        rpar = Leaf(token.RPAR, \')\')\n        insert_str_child(rpar)\n        ```\n\n        After which point, it follows that `string_leaf.parent is None` and\n        the node `N` now has the following structure:\n\n        Node(\n            expr_stmt, [\n                Leaf(NAME, \'x\'),\n                Leaf(EQUAL, \'=\'),\n                Leaf(LPAR, \'(\'),\n                Leaf(STRING, \'"bar"\'),\n                Leaf(RPAR, \')\'),\n            ]\n        )\n    '
    string_parent = string_leaf.parent
    string_child_idx = string_leaf.remove()

    def insert_str_child(child: LN) -> None:
        nonlocal string_child_idx
        assert (string_parent is not None)
        assert (string_child_idx is not None)
        string_parent.insert_child(string_child_idx, child)
        string_child_idx += 1
    return insert_str_child

def has_triple_quotes(string):
    '\n    Returns:\n        True iff @string starts with three quotation characters.\n    '
    raw_string = string.lstrip(STRING_PREFIX_CHARS)
    return (raw_string[:3] in {'"""', "'''"})

def parent_type(node):
    '\n    Returns:\n        @node.parent.type, if @node is not None and has a parent.\n            OR\n        None, otherwise.\n    '
    if ((node is None) or (node.parent is None)):
        return None
    return node.parent.type

def is_empty_par(leaf):
    return (is_empty_lpar(leaf) or is_empty_rpar(leaf))

def is_empty_lpar(leaf):
    return ((leaf.type == token.LPAR) and (leaf.value == ''))

def is_empty_rpar(leaf):
    return ((leaf.type == token.RPAR) and (leaf.value == ''))

def is_valid_index_factory(seq):
    '\n    Examples:\n        ```\n        my_list = [1, 2, 3]\n\n        is_valid_index = is_valid_index_factory(my_list)\n\n        assert is_valid_index(0)\n        assert is_valid_index(2)\n\n        assert not is_valid_index(3)\n        assert not is_valid_index(-1)\n        ```\n    '

    def is_valid_index(idx: int) -> bool:
        '\n        Returns:\n            True iff @idx is positive AND seq[@idx] does NOT raise an\n            IndexError.\n        '
        return (0 <= idx < len(seq))
    return is_valid_index

def line_to_string(line):
    'Returns the string representation of @line.\n\n    WARNING: This is known to be computationally expensive.\n    '
    return str(line).strip('\n')

def append_leaves(new_line, old_line, leaves, preformatted=False):
    '\n    Append leaves (taken from @old_line) to @new_line, making sure to fix the\n    underlying Node structure where appropriate.\n\n    All of the leaves in @leaves are duplicated. The duplicates are then\n    appended to @new_line and used to replace their originals in the underlying\n    Node structure. Any comments attached to the old leaves are reattached to\n    the new leaves.\n\n    Pre-conditions:\n        set(@leaves) is a subset of set(@old_line.leaves).\n    '
    for old_leaf in leaves:
        new_leaf = Leaf(old_leaf.type, old_leaf.value)
        replace_child(old_leaf, new_leaf)
        new_line.append(new_leaf, preformatted=preformatted)
        for comment_leaf in old_line.comments_after(old_leaf):
            new_line.append(comment_leaf, preformatted=True)

def replace_child(old_child, new_child):
    "\n    Side Effects:\n        * If @old_child.parent is set, replace @old_child with @new_child in\n        @old_child's underlying Node structure.\n            OR\n        * Otherwise, this function does nothing.\n    "
    parent = old_child.parent
    if (not parent):
        return
    child_idx = old_child.remove()
    if (child_idx is not None):
        parent.insert_child(child_idx, new_child)

def get_string_prefix(string):
    "\n    Pre-conditions:\n        * assert_is_leaf_string(@string)\n\n    Returns:\n        @string's prefix (e.g. '', 'r', 'f', or 'rf').\n    "
    assert_is_leaf_string(string)
    prefix = ''
    prefix_idx = 0
    while (string[prefix_idx] in STRING_PREFIX_CHARS):
        prefix += string[prefix_idx].lower()
        prefix_idx += 1
    return prefix

def assert_is_leaf_string(string):
    '\n    Checks the pre-condition that @string has the format that you would expect\n    of `leaf.value` where `leaf` is some Leaf such that `leaf.type ==\n    token.STRING`. A more precise description of the pre-conditions that are\n    checked are listed below.\n\n    Pre-conditions:\n        * @string starts with either \', ", <prefix>\', or <prefix>" where\n        `set(<prefix>)` is some subset of `set(STRING_PREFIX_CHARS)`.\n        * @string ends with a quote character (\' or ").\n\n    Raises:\n        AssertionError(...) if the pre-conditions listed above are not\n        satisfied.\n    '
    dquote_idx = string.find('"')
    squote_idx = string.find("'")
    if ((- 1) in [dquote_idx, squote_idx]):
        quote_idx = max(dquote_idx, squote_idx)
    else:
        quote_idx = min(squote_idx, dquote_idx)
    assert (0 <= quote_idx < (len(string) - 1)), f"""{string!r} is missing a starting quote character (' or ")."""
    assert (string[(- 1)] in ("'", '"')), f"""{string!r} is missing an ending quote character (' or ")."""
    assert set(string[:quote_idx]).issubset(set(STRING_PREFIX_CHARS)), f'{set(string[:quote_idx])} is NOT a subset of {set(STRING_PREFIX_CHARS)}.'

def left_hand_split(line, _features=()):
    'Split line into many lines, starting with the first matching bracket pair.\n\n    Note: this usually looks weird, only use this for function definitions.\n    Prefer RHS otherwise.  This is why this function is not symmetrical with\n    :func:`right_hand_split` which also handles optional parentheses.\n    '
    tail_leaves: List[Leaf] = []
    body_leaves: List[Leaf] = []
    head_leaves: List[Leaf] = []
    current_leaves = head_leaves
    matching_bracket: Optional[Leaf] = None
    for leaf in line.leaves:
        if ((current_leaves is body_leaves) and (leaf.type in CLOSING_BRACKETS) and (leaf.opening_bracket is matching_bracket)):
            current_leaves = (tail_leaves if body_leaves else head_leaves)
        current_leaves.append(leaf)
        if (current_leaves is head_leaves):
            if (leaf.type in OPENING_BRACKETS):
                matching_bracket = leaf
                current_leaves = body_leaves
    if (not matching_bracket):
        raise CannotSplit('No brackets found')
    head = bracket_split_build_line(head_leaves, line, matching_bracket)
    body = bracket_split_build_line(body_leaves, line, matching_bracket, is_body=True)
    tail = bracket_split_build_line(tail_leaves, line, matching_bracket)
    bracket_split_succeeded_or_raise(head, body, tail)
    for result in (head, body, tail):
        if result:
            (yield result)

def right_hand_split(line, line_length, features=(), omit=()):
    "Split line into many lines, starting with the last matching bracket pair.\n\n    If the split was by optional parentheses, attempt splitting without them, too.\n    `omit` is a collection of closing bracket IDs that shouldn't be considered for\n    this split.\n\n    Note: running this function modifies `bracket_depth` on the leaves of `line`.\n    "
    tail_leaves: List[Leaf] = []
    body_leaves: List[Leaf] = []
    head_leaves: List[Leaf] = []
    current_leaves = tail_leaves
    opening_bracket: Optional[Leaf] = None
    closing_bracket: Optional[Leaf] = None
    for leaf in reversed(line.leaves):
        if (current_leaves is body_leaves):
            if (leaf is opening_bracket):
                current_leaves = (head_leaves if body_leaves else tail_leaves)
        current_leaves.append(leaf)
        if (current_leaves is tail_leaves):
            if ((leaf.type in CLOSING_BRACKETS) and (id(leaf) not in omit)):
                opening_bracket = leaf.opening_bracket
                closing_bracket = leaf
                current_leaves = body_leaves
    if (not (opening_bracket and closing_bracket and head_leaves)):
        raise CannotSplit('No brackets found')
    tail_leaves.reverse()
    body_leaves.reverse()
    head_leaves.reverse()
    head = bracket_split_build_line(head_leaves, line, opening_bracket)
    body = bracket_split_build_line(body_leaves, line, opening_bracket, is_body=True)
    tail = bracket_split_build_line(tail_leaves, line, opening_bracket)
    bracket_split_succeeded_or_raise(head, body, tail)
    if ((Feature.FORCE_OPTIONAL_PARENTHESES not in features) and (opening_bracket.type == token.LPAR) and (not opening_bracket.value) and (closing_bracket.type == token.RPAR) and (not closing_bracket.value) and (not line.is_import) and (not body.contains_standalone_comments(0)) and can_omit_invisible_parens(body, line_length, omit_on_explode=omit)):
        omit = {id(closing_bracket), *omit}
        try:
            (yield from right_hand_split(line, line_length, features=features, omit=omit))
            return
        except CannotSplit:
            if (not (can_be_split(body) or is_line_short_enough(body, line_length=line_length))):
                raise CannotSplit("Splitting failed, body is still too long and can't be split.")
            elif (head.contains_multiline_strings() or tail.contains_multiline_strings()):
                raise CannotSplit('The current optional pair of parentheses is bound to fail to satisfy the splitting algorithm because the head or the tail contains multiline strings which by definition never fit one line.')
    ensure_visible(opening_bracket)
    ensure_visible(closing_bracket)
    for result in (head, body, tail):
        if result:
            (yield result)

def bracket_split_succeeded_or_raise(head, body, tail):
    "Raise :exc:`CannotSplit` if the last left- or right-hand split failed.\n\n    Do nothing otherwise.\n\n    A left- or right-hand split is based on a pair of brackets. Content before\n    (and including) the opening bracket is left on one line, content inside the\n    brackets is put on a separate line, and finally content starting with and\n    following the closing bracket is put on a separate line.\n\n    Those are called `head`, `body`, and `tail`, respectively. If the split\n    produced the same line (all content in `head`) or ended up with an empty `body`\n    and the `tail` is just the closing bracket, then it's considered failed.\n    "
    tail_len = len(str(tail).strip())
    if (not body):
        if (tail_len == 0):
            raise CannotSplit('Splitting brackets produced the same line')
        elif (tail_len < 3):
            raise CannotSplit(f'Splitting brackets on an empty body to save {tail_len} characters is not worth it')

def bracket_split_build_line(leaves, original, opening_bracket, *, is_body=False):
    "Return a new line with given `leaves` and respective comments from `original`.\n\n    If `is_body` is True, the result line is one-indented inside brackets and as such\n    has its first leaf's prefix normalized and a trailing comma added when expected.\n    "
    result = Line(depth=original.depth)
    if is_body:
        result.inside_brackets = True
        result.depth += 1
        if leaves:
            normalize_prefix(leaves[0], inside_brackets=True)
            no_commas = (original.is_def and (opening_bracket.value == '(') and (not any(((leaf.type == token.COMMA) for leaf in leaves))))
            if (original.is_import or no_commas):
                for i in range((len(leaves) - 1), (- 1), (- 1)):
                    if (leaves[i].type == STANDALONE_COMMENT):
                        continue
                    if (leaves[i].type != token.COMMA):
                        new_comma = Leaf(token.COMMA, ',')
                        leaves.insert((i + 1), new_comma)
                    break
    for leaf in leaves:
        result.append(leaf, preformatted=True)
        for comment_after in original.comments_after(leaf):
            result.append(comment_after, preformatted=True)
    if (is_body and should_split_body_explode(result, opening_bracket)):
        result.should_explode = True
    return result

def dont_increase_indentation(split_func):
    'Normalize prefix of the first leaf in every line returned by `split_func`.\n\n    This is a decorator over relevant split functions.\n    '

    @wraps(split_func)
    def split_wrapper(line: Line, features: Collection[Feature]=()) -> Iterator[Line]:
        for line in split_func(line, features):
            normalize_prefix(line.leaves[0], inside_brackets=True)
            (yield line)
    return split_wrapper

@dont_increase_indentation
def delimiter_split(line, features=()):
    'Split according to delimiters of the highest priority.\n\n    If the appropriate Features are given, the split will add trailing commas\n    also in function signatures and calls that contain `*` and `**`.\n    '
    try:
        last_leaf = line.leaves[(- 1)]
    except IndexError:
        raise CannotSplit('Line empty')
    bt = line.bracket_tracker
    try:
        delimiter_priority = bt.max_delimiter_priority(exclude={id(last_leaf)})
    except ValueError:
        raise CannotSplit('No delimiters found')
    if (delimiter_priority == DOT_PRIORITY):
        if (bt.delimiter_count_with_priority(delimiter_priority) == 1):
            raise CannotSplit('Splitting a single attribute from its owner looks wrong')
    current_line = Line(depth=line.depth, inside_brackets=line.inside_brackets)
    lowest_depth = sys.maxsize
    trailing_comma_safe = True

    def append_to_line(leaf: Leaf) -> Iterator[Line]:
        'Append `leaf` to current line or to new line if appending impossible.'
        nonlocal current_line
        try:
            current_line.append_safe(leaf, preformatted=True)
        except ValueError:
            (yield current_line)
            current_line = Line(depth=line.depth, inside_brackets=line.inside_brackets)
            current_line.append(leaf)
    for leaf in line.leaves:
        (yield from append_to_line(leaf))
        for comment_after in line.comments_after(leaf):
            (yield from append_to_line(comment_after))
        lowest_depth = min(lowest_depth, leaf.bracket_depth)
        if (leaf.bracket_depth == lowest_depth):
            if is_vararg(leaf, within={syms.typedargslist}):
                trailing_comma_safe = (trailing_comma_safe and (Feature.TRAILING_COMMA_IN_DEF in features))
            elif is_vararg(leaf, within={syms.arglist, syms.argument}):
                trailing_comma_safe = (trailing_comma_safe and (Feature.TRAILING_COMMA_IN_CALL in features))
        leaf_priority = bt.delimiters.get(id(leaf))
        if (leaf_priority == delimiter_priority):
            (yield current_line)
            current_line = Line(depth=line.depth, inside_brackets=line.inside_brackets)
    if current_line:
        if (trailing_comma_safe and (delimiter_priority == COMMA_PRIORITY) and (current_line.leaves[(- 1)].type != token.COMMA) and (current_line.leaves[(- 1)].type != STANDALONE_COMMENT)):
            new_comma = Leaf(token.COMMA, ',')
            current_line.append(new_comma)
        (yield current_line)

@dont_increase_indentation
def standalone_comment_split(line, features=()):
    'Split standalone comments from the rest of the line.'
    if (not line.contains_standalone_comments(0)):
        raise CannotSplit('Line does not have any standalone comments')
    current_line = Line(depth=line.depth, inside_brackets=line.inside_brackets)

    def append_to_line(leaf: Leaf) -> Iterator[Line]:
        'Append `leaf` to current line or to new line if appending impossible.'
        nonlocal current_line
        try:
            current_line.append_safe(leaf, preformatted=True)
        except ValueError:
            (yield current_line)
            current_line = Line(depth=line.depth, inside_brackets=line.inside_brackets)
            current_line.append(leaf)
    for leaf in line.leaves:
        (yield from append_to_line(leaf))
        for comment_after in line.comments_after(leaf):
            (yield from append_to_line(comment_after))
    if current_line:
        (yield current_line)

def is_import(leaf):
    'Return True if the given leaf starts an import statement.'
    p = leaf.parent
    t = leaf.type
    v = leaf.value
    return bool(((t == token.NAME) and (((v == 'import') and p and (p.type == syms.import_name)) or ((v == 'from') and p and (p.type == syms.import_from)))))

def is_type_comment(leaf, suffix=''):
    'Return True if the given leaf is a special comment.\n    Only returns true for type comments for now.'
    t = leaf.type
    v = leaf.value
    return ((t in {token.COMMENT, STANDALONE_COMMENT}) and v.startswith(('# type:' + suffix)))

def normalize_prefix(leaf, *, inside_brackets):
    "Leave existing extra newlines if not `inside_brackets`. Remove everything\n    else.\n\n    Note: don't use backslashes for formatting or you'll lose your voting rights.\n    "
    if (not inside_brackets):
        spl = leaf.prefix.split('#')
        if ('\\' not in spl[0]):
            nl_count = spl[(- 1)].count('\n')
            if (len(spl) > 1):
                nl_count -= 1
            leaf.prefix = ('\n' * nl_count)
            return
    leaf.prefix = ''

def normalize_string_prefix(leaf, remove_u_prefix=False):
    'Make all string prefixes lowercase.\n\n    If remove_u_prefix is given, also removes any u prefix from the string.\n\n    Note: Mutates its argument.\n    '
    match = re.match((('^([' + STRING_PREFIX_CHARS) + ']*)(.*)$'), leaf.value, re.DOTALL)
    assert (match is not None), f'failed to match string {leaf.value!r}'
    orig_prefix = match.group(1)
    new_prefix = orig_prefix.replace('F', 'f').replace('B', 'b').replace('U', 'u')
    if remove_u_prefix:
        new_prefix = new_prefix.replace('u', '')
    leaf.value = f'{new_prefix}{match.group(2)}'

def normalize_string_quotes(leaf):
    "Prefer double quotes but only if it doesn't cause more escaping.\n\n    Adds or removes backslashes as appropriate. Doesn't parse and fix\n    strings nested in f-strings (yet).\n\n    Note: Mutates its argument.\n    "
    value = leaf.value.lstrip(STRING_PREFIX_CHARS)
    if (value[:3] == '"""'):
        return
    elif (value[:3] == "'''"):
        orig_quote = "'''"
        new_quote = '"""'
    elif (value[0] == '"'):
        orig_quote = '"'
        new_quote = "'"
    else:
        orig_quote = "'"
        new_quote = '"'
    first_quote_pos = leaf.value.find(orig_quote)
    if (first_quote_pos == (- 1)):
        return
    prefix = leaf.value[:first_quote_pos]
    unescaped_new_quote = re.compile(f'(([^\\]|^)(\\\\)*){new_quote}')
    escaped_new_quote = re.compile(f'([^\\]|^)\\((?:\\\\)*){new_quote}')
    escaped_orig_quote = re.compile(f'([^\\]|^)\\((?:\\\\)*){orig_quote}')
    body = leaf.value[(first_quote_pos + len(orig_quote)):(- len(orig_quote))]
    if ('r' in prefix.casefold()):
        if unescaped_new_quote.search(body):
            return
        new_body = body
    else:
        new_body = sub_twice(escaped_new_quote, f'\1\2{new_quote}', body)
        if (body != new_body):
            body = new_body
            leaf.value = f'{prefix}{orig_quote}{body}{orig_quote}'
        new_body = sub_twice(escaped_orig_quote, f'\1\2{orig_quote}', new_body)
        new_body = sub_twice(unescaped_new_quote, f'\1\\{new_quote}', new_body)
    if ('f' in prefix.casefold()):
        matches = re.findall('\n            (?:[^{]|^)\\{  # start of the string or a non-{ followed by a single {\n                ([^{].*?)  # contents of the brackets except if begins with {{\n            \\}(?:[^}]|$)  # A } followed by end of the string or a non-}\n            ', new_body, re.VERBOSE)
        for m in matches:
            if ('\\' in str(m)):
                return
    if ((new_quote == '"""') and (new_body[(- 1):] == '"')):
        new_body = (new_body[:(- 1)] + '\\"')
    orig_escape_count = body.count('\\')
    new_escape_count = new_body.count('\\')
    if (new_escape_count > orig_escape_count):
        return
    if ((new_escape_count == orig_escape_count) and (orig_quote == '"')):
        return
    leaf.value = f'{prefix}{new_quote}{new_body}{new_quote}'

def normalize_numeric_literal(leaf):
    'Normalizes numeric (float, int, and complex) literals.\n\n    All letters used in the representation are normalized to lowercase (except\n    in Python 2 long literals).\n    '
    text = leaf.value.lower()
    if text.startswith(('0o', '0b')):
        pass
    elif text.startswith('0x'):
        text = format_hex(text)
    elif ('e' in text):
        text = format_scientific_notation(text)
    elif text.endswith(('j', 'l')):
        text = format_long_or_complex_number(text)
    else:
        text = format_float_or_int_string(text)
    leaf.value = text

def format_hex(text):
    '\n    Formats a hexadecimal string like "0x12b3"\n\n    Uses lowercase because of similarity between "B" and "8", which\n    can cause security issues.\n    see: https://github.com/psf/black/issues/1692\n    '
    (before, after) = (text[:2], text[2:])
    return f'{before}{after.lower()}'

def format_scientific_notation(text):
    'Formats a numeric string utilizing scentific notation'
    (before, after) = text.split('e')
    sign = ''
    if after.startswith('-'):
        after = after[1:]
        sign = '-'
    elif after.startswith('+'):
        after = after[1:]
    before = format_float_or_int_string(before)
    return f'{before}e{sign}{after}'

def format_long_or_complex_number(text):
    'Formats a long or complex string like `10L` or `10j`'
    number = text[:(- 1)]
    suffix = text[(- 1)]
    if (suffix == 'l'):
        suffix = 'L'
    return f'{format_float_or_int_string(number)}{suffix}'

def format_float_or_int_string(text):
    'Formats a float string like "1.0".'
    if ('.' not in text):
        return text
    (before, after) = text.split('.')
    return f'{(before or 0)}.{(after or 0)}'

def normalize_invisible_parens(node, parens_after):
    'Make existing optional parentheses invisible or create new ones.\n\n    `parens_after` is a set of string leaf values immediately after which parens\n    should be put.\n\n    Standardizes on visible parentheses for single-element tuples, and keeps\n    existing visible parentheses for other tuples and generator expressions.\n    '
    for pc in list_comments(node.prefix, is_endmarker=False):
        if (pc.value in FMT_OFF):
            return
    check_lpar = False
    for (index, child) in enumerate(list(node.children)):
        if (isinstance(child, Node) and (child.type == syms.annassign)):
            normalize_invisible_parens(child, parens_after=parens_after)
        if ((index == 0) and isinstance(child, Node) and (child.type == syms.testlist_star_expr)):
            check_lpar = True
        if check_lpar:
            if is_walrus_assignment(child):
                pass
            elif (child.type == syms.atom):
                if maybe_make_parens_invisible_in_atom(child, parent=node):
                    wrap_in_parentheses(node, child, visible=False)
            elif is_one_tuple(child):
                wrap_in_parentheses(node, child, visible=True)
            elif (node.type == syms.import_from):
                if (child.type == token.LPAR):
                    child.value = ''
                    node.children[(- 1)].value = ''
                elif (child.type != token.STAR):
                    node.insert_child(index, Leaf(token.LPAR, ''))
                    node.append_child(Leaf(token.RPAR, ''))
                break
            elif (not (isinstance(child, Leaf) and is_multiline_string(child))):
                wrap_in_parentheses(node, child, visible=False)
        check_lpar = (isinstance(child, Leaf) and (child.value in parens_after))

def normalize_fmt_off(node):
    'Convert content between `# fmt: off`/`# fmt: on` into standalone comments.'
    try_again = True
    while try_again:
        try_again = convert_one_fmt_off_pair(node)

def convert_one_fmt_off_pair(node):
    'Convert content of a single `# fmt: off`/`# fmt: on` into a standalone comment.\n\n    Returns True if a pair was converted.\n    '
    for leaf in node.leaves():
        previous_consumed = 0
        for comment in list_comments(leaf.prefix, is_endmarker=False):
            if (comment.value in FMT_OFF):
                if (comment.type != STANDALONE_COMMENT):
                    prev = preceding_leaf(leaf)
                    if (prev and (prev.type not in WHITESPACE)):
                        continue
                ignored_nodes = list(generate_ignored_nodes(leaf))
                if (not ignored_nodes):
                    continue
                first = ignored_nodes[0]
                parent = first.parent
                prefix = first.prefix
                first.prefix = prefix[comment.consumed:]
                hidden_value = ((comment.value + '\n') + ''.join((str(n) for n in ignored_nodes)))
                if hidden_value.endswith('\n'):
                    hidden_value = hidden_value[:(- 1)]
                first_idx: Optional[int] = None
                for ignored in ignored_nodes:
                    index = ignored.remove()
                    if (first_idx is None):
                        first_idx = index
                assert (parent is not None), 'INTERNAL ERROR: fmt: on/off handling (1)'
                assert (first_idx is not None), 'INTERNAL ERROR: fmt: on/off handling (2)'
                parent.insert_child(first_idx, Leaf(STANDALONE_COMMENT, hidden_value, prefix=(prefix[:previous_consumed] + ('\n' * comment.newlines))))
                return True
            previous_consumed = comment.consumed
    return False

def generate_ignored_nodes(leaf):
    'Starting from the container of `leaf`, generate all leaves until `# fmt: on`.\n\n    Stops at the end of the block.\n    '
    container: Optional[LN] = container_of(leaf)
    while ((container is not None) and (container.type != token.ENDMARKER)):
        if is_fmt_on(container):
            return
        if contains_fmt_on_at_column(container, leaf.column):
            for child in container.children:
                if contains_fmt_on_at_column(child, leaf.column):
                    return
                (yield child)
        else:
            (yield container)
            container = container.next_sibling

def is_fmt_on(container):
    'Determine whether formatting is switched on within a container.\n    Determined by whether the last `# fmt:` comment is `on` or `off`.\n    '
    fmt_on = False
    for comment in list_comments(container.prefix, is_endmarker=False):
        if (comment.value in FMT_ON):
            fmt_on = True
        elif (comment.value in FMT_OFF):
            fmt_on = False
    return fmt_on

def contains_fmt_on_at_column(container, column):
    'Determine if children at a given column have formatting switched on.'
    for child in container.children:
        if ((isinstance(child, Node) and (first_leaf_column(child) == column)) or (isinstance(child, Leaf) and (child.column == column))):
            if is_fmt_on(child):
                return True
    return False

def first_leaf_column(node):
    'Returns the column of the first leaf child of a node.'
    for child in node.children:
        if isinstance(child, Leaf):
            return child.column
    return None

def maybe_make_parens_invisible_in_atom(node, parent):
    "If it's safe, make the parens in the atom `node` invisible, recursively.\n    Additionally, remove repeated, adjacent invisible parens from the atom `node`\n    as they are redundant.\n\n    Returns whether the node should itself be wrapped in invisible parentheses.\n\n    "
    if ((node.type != syms.atom) or is_empty_tuple(node) or is_one_tuple(node) or (is_yield(node) and (parent.type != syms.expr_stmt)) or (max_delimiter_priority_in_atom(node) >= COMMA_PRIORITY)):
        return False
    first = node.children[0]
    last = node.children[(- 1)]
    if ((first.type == token.LPAR) and (last.type == token.RPAR)):
        middle = node.children[1]
        first.value = ''
        last.value = ''
        maybe_make_parens_invisible_in_atom(middle, parent=parent)
        if is_atom_with_invisible_parens(middle):
            middle.replace(middle.children[1])
        return False
    return True

def is_atom_with_invisible_parens(node):
    "Given a `LN`, determines whether it's an atom `node` with invisible\n    parens. Useful in dedupe-ing and normalizing parens.\n    "
    if (isinstance(node, Leaf) or (node.type != syms.atom)):
        return False
    (first, last) = (node.children[0], node.children[(- 1)])
    return (isinstance(first, Leaf) and (first.type == token.LPAR) and (first.value == '') and isinstance(last, Leaf) and (last.type == token.RPAR) and (last.value == ''))

def is_empty_tuple(node):
    'Return True if `node` holds an empty tuple.'
    return ((node.type == syms.atom) and (len(node.children) == 2) and (node.children[0].type == token.LPAR) and (node.children[1].type == token.RPAR))

def unwrap_singleton_parenthesis(node):
    'Returns `wrapped` if `node` is of the shape ( wrapped ).\n\n    Parenthesis can be optional. Returns None otherwise'
    if (len(node.children) != 3):
        return None
    (lpar, wrapped, rpar) = node.children
    if (not ((lpar.type == token.LPAR) and (rpar.type == token.RPAR))):
        return None
    return wrapped

def wrap_in_parentheses(parent, child, *, visible=True):
    'Wrap `child` in parentheses.\n\n    This replaces `child` with an atom holding the parentheses and the old\n    child.  That requires moving the prefix.\n\n    If `visible` is False, the leaves will be valueless (and thus invisible).\n    '
    lpar = Leaf(token.LPAR, ('(' if visible else ''))
    rpar = Leaf(token.RPAR, (')' if visible else ''))
    prefix = child.prefix
    child.prefix = ''
    index = (child.remove() or 0)
    new_child = Node(syms.atom, [lpar, child, rpar])
    new_child.prefix = prefix
    parent.insert_child(index, new_child)

def is_one_tuple(node):
    'Return True if `node` holds a tuple with one element, with or without parens.'
    if (node.type == syms.atom):
        gexp = unwrap_singleton_parenthesis(node)
        if ((gexp is None) or (gexp.type != syms.testlist_gexp)):
            return False
        return ((len(gexp.children) == 2) and (gexp.children[1].type == token.COMMA))
    return ((node.type in IMPLICIT_TUPLE) and (len(node.children) == 2) and (node.children[1].type == token.COMMA))

def is_walrus_assignment(node):
    'Return True iff `node` is of the shape ( test := test )'
    inner = unwrap_singleton_parenthesis(node)
    return ((inner is not None) and (inner.type == syms.namedexpr_test))

def is_simple_decorator_trailer(node, last=False):
    'Return True iff `node` is a trailer valid in a simple decorator'
    return ((node.type == syms.trailer) and (((len(node.children) == 2) and (node.children[0].type == token.DOT) and (node.children[1].type == token.NAME)) or (last and (len(node.children) == 3) and (node.children[0].type == token.LPAR) and (node.children[2].type == token.RPAR))))

def is_simple_decorator_expression(node):
    "Return True iff `node` could be a 'dotted name' decorator\n\n    This function takes the node of the 'namedexpr_test' of the new decorator\n    grammar and test if it would be valid under the old decorator grammar.\n\n    The old grammar was: decorator: @ dotted_name [arguments] NEWLINE\n    The new grammar is : decorator: @ namedexpr_test NEWLINE\n    "
    if (node.type == token.NAME):
        return True
    if (node.type == syms.power):
        if node.children:
            return ((node.children[0].type == token.NAME) and all(map(is_simple_decorator_trailer, node.children[1:(- 1)])) and ((len(node.children) < 2) or is_simple_decorator_trailer(node.children[(- 1)], last=True)))
    return False

def is_yield(node):
    'Return True if `node` holds a `yield` or `yield from` expression.'
    if (node.type == syms.yield_expr):
        return True
    if ((node.type == token.NAME) and (node.value == 'yield')):
        return True
    if (node.type != syms.atom):
        return False
    if (len(node.children) != 3):
        return False
    (lpar, expr, rpar) = node.children
    if ((lpar.type == token.LPAR) and (rpar.type == token.RPAR)):
        return is_yield(expr)
    return False

def is_vararg(leaf, within):
    'Return True if `leaf` is a star or double star in a vararg or kwarg.\n\n    If `within` includes VARARGS_PARENTS, this applies to function signatures.\n    If `within` includes UNPACKING_PARENTS, it applies to right hand-side\n    extended iterable unpacking (PEP 3132) and additional unpacking\n    generalizations (PEP 448).\n    '
    if ((leaf.type not in VARARGS_SPECIALS) or (not leaf.parent)):
        return False
    p = leaf.parent
    if (p.type == syms.star_expr):
        if (not p.parent):
            return False
        p = p.parent
    return (p.type in within)

def is_multiline_string(leaf):
    'Return True if `leaf` is a multiline string that actually spans many lines.'
    return (has_triple_quotes(leaf.value) and ('\n' in leaf.value))

def is_stub_suite(node):
    'Return True if `node` is a suite with a stub body.'
    if ((len(node.children) != 4) or (node.children[0].type != token.NEWLINE) or (node.children[1].type != token.INDENT) or (node.children[3].type != token.DEDENT)):
        return False
    return is_stub_body(node.children[2])

def is_stub_body(node):
    'Return True if `node` is a simple statement containing an ellipsis.'
    if ((not isinstance(node, Node)) or (node.type != syms.simple_stmt)):
        return False
    if (len(node.children) != 2):
        return False
    child = node.children[0]
    return ((child.type == syms.atom) and (len(child.children) == 3) and all(((leaf == Leaf(token.DOT, '.')) for leaf in child.children)))

def max_delimiter_priority_in_atom(node):
    "Return maximum delimiter priority inside `node`.\n\n    This is specific to atoms with contents contained in a pair of parentheses.\n    If `node` isn't an atom or there are no enclosing parentheses, returns 0.\n    "
    if (node.type != syms.atom):
        return 0
    first = node.children[0]
    last = node.children[(- 1)]
    if (not ((first.type == token.LPAR) and (last.type == token.RPAR))):
        return 0
    bt = BracketTracker()
    for c in node.children[1:(- 1)]:
        if isinstance(c, Leaf):
            bt.mark(c)
        else:
            for leaf in c.leaves():
                bt.mark(leaf)
    try:
        return bt.max_delimiter_priority()
    except ValueError:
        return 0

def ensure_visible(leaf):
    'Make sure parentheses are visible.\n\n    They could be invisible as part of some statements (see\n    :func:`normalize_invisible_parens` and :func:`visit_import_from`).\n    '
    if (leaf.type == token.LPAR):
        leaf.value = '('
    elif (leaf.type == token.RPAR):
        leaf.value = ')'

def should_split_body_explode(line, opening_bracket):
    'Should `line` be immediately split with `delimiter_split()` after RHS?'
    if (not (opening_bracket.parent and (opening_bracket.value in '[{('))):
        return False
    exclude = set()
    trailing_comma = False
    try:
        last_leaf = line.leaves[(- 1)]
        if (last_leaf.type == token.COMMA):
            trailing_comma = True
            exclude.add(id(last_leaf))
        max_priority = line.bracket_tracker.max_delimiter_priority(exclude=exclude)
    except (IndexError, ValueError):
        return False
    return ((max_priority == COMMA_PRIORITY) and (trailing_comma or (opening_bracket.parent.type in {syms.atom, syms.import_from})))

def is_one_tuple_between(opening, closing, leaves):
    'Return True if content between `opening` and `closing` looks like a one-tuple.'
    if ((opening.type != token.LPAR) and (closing.type != token.RPAR)):
        return False
    depth = (closing.bracket_depth + 1)
    for (_opening_index, leaf) in enumerate(leaves):
        if (leaf is opening):
            break
    else:
        raise LookupError('Opening paren not found in `leaves`')
    commas = 0
    _opening_index += 1
    for leaf in leaves[_opening_index:]:
        if (leaf is closing):
            break
        bracket_depth = leaf.bracket_depth
        if ((bracket_depth == depth) and (leaf.type == token.COMMA)):
            commas += 1
            if (leaf.parent and (leaf.parent.type in {syms.arglist, syms.typedargslist})):
                commas += 1
                break
    return (commas < 2)

def get_features_used(node):
    'Return a set of (relatively) new Python features used in this file.\n\n    Currently looking for:\n    - f-strings;\n    - underscores in numeric literals;\n    - trailing commas after * or ** in function signatures and calls;\n    - positional only arguments in function signatures and lambdas;\n    - assignment expression;\n    - relaxed decorator syntax;\n    '
    features: Set[Feature] = set()
    for n in node.pre_order():
        if (n.type == token.STRING):
            value_head = n.value[:2]
            if (value_head in {'f"', 'F"', "f'", "F'", 'rf', 'fr', 'RF', 'FR'}):
                features.add(Feature.F_STRINGS)
        elif (n.type == token.NUMBER):
            if ('_' in n.value):
                features.add(Feature.NUMERIC_UNDERSCORES)
        elif (n.type == token.SLASH):
            if (n.parent and (n.parent.type in {syms.typedargslist, syms.arglist})):
                features.add(Feature.POS_ONLY_ARGUMENTS)
        elif (n.type == token.COLONEQUAL):
            features.add(Feature.ASSIGNMENT_EXPRESSIONS)
        elif (n.type == syms.decorator):
            if ((len(n.children) > 1) and (not is_simple_decorator_expression(n.children[1]))):
                features.add(Feature.RELAXED_DECORATORS)
        elif ((n.type in {syms.typedargslist, syms.arglist}) and n.children and (n.children[(- 1)].type == token.COMMA)):
            if (n.type == syms.typedargslist):
                feature = Feature.TRAILING_COMMA_IN_DEF
            else:
                feature = Feature.TRAILING_COMMA_IN_CALL
            for ch in n.children:
                if (ch.type in STARS):
                    features.add(feature)
                if (ch.type == syms.argument):
                    for argch in ch.children:
                        if (argch.type in STARS):
                            features.add(feature)
    return features

def detect_target_versions(node):
    'Detect the version to target based on the nodes used.'
    features = get_features_used(node)
    return {version for version in TargetVersion if (features <= VERSION_TO_FEATURES[version])}

def generate_trailers_to_omit(line, line_length):
    'Generate sets of closing bracket IDs that should be omitted in a RHS.\n\n    Brackets can be omitted if the entire trailer up to and including\n    a preceding closing bracket fits in one line.\n\n    Yielded sets are cumulative (contain results of previous yields, too).  First\n    set is empty, unless the line should explode, in which case bracket pairs until\n    the one that needs to explode are omitted.\n    '
    omit: Set[LeafID] = set()
    if (not line.should_explode):
        (yield omit)
    length = (4 * line.depth)
    opening_bracket: Optional[Leaf] = None
    closing_bracket: Optional[Leaf] = None
    inner_brackets: Set[LeafID] = set()
    for (index, leaf, leaf_length) in enumerate_with_length(line, reversed=True):
        length += leaf_length
        if (length > line_length):
            break
        has_inline_comment = (leaf_length > (len(leaf.value) + len(leaf.prefix)))
        if ((leaf.type == STANDALONE_COMMENT) or has_inline_comment):
            break
        if opening_bracket:
            if (leaf is opening_bracket):
                opening_bracket = None
            elif (leaf.type in CLOSING_BRACKETS):
                prev = (line.leaves[(index - 1)] if (index > 0) else None)
                if (line.should_explode and prev and (prev.type == token.COMMA) and (not is_one_tuple_between(leaf.opening_bracket, leaf, line.leaves))):
                    break
                inner_brackets.add(id(leaf))
        elif (leaf.type in CLOSING_BRACKETS):
            prev = (line.leaves[(index - 1)] if (index > 0) else None)
            if (prev and (prev.type in OPENING_BRACKETS)):
                inner_brackets.add(id(leaf))
                continue
            if closing_bracket:
                omit.add(id(closing_bracket))
                omit.update(inner_brackets)
                inner_brackets.clear()
                (yield omit)
            if (line.should_explode and prev and (prev.type == token.COMMA) and (not is_one_tuple_between(leaf.opening_bracket, leaf, line.leaves))):
                break
            if leaf.value:
                opening_bracket = leaf.opening_bracket
                closing_bracket = leaf

def get_future_imports(node):
    'Return a set of __future__ imports in the file.'
    imports: Set[str] = set()

    def get_imports_from_children(children: List[LN]) -> Generator[(str, None, None)]:
        for child in children:
            if isinstance(child, Leaf):
                if (child.type == token.NAME):
                    (yield child.value)
            elif (child.type == syms.import_as_name):
                orig_name = child.children[0]
                assert isinstance(orig_name, Leaf), 'Invalid syntax parsing imports'
                assert (orig_name.type == token.NAME), 'Invalid syntax parsing imports'
                (yield orig_name.value)
            elif (child.type == syms.import_as_names):
                (yield from get_imports_from_children(child.children))
            else:
                raise AssertionError('Invalid syntax parsing imports')
    for child in node.children:
        if (child.type != syms.simple_stmt):
            break
        first_child = child.children[0]
        if isinstance(first_child, Leaf):
            if ((len(child.children) == 2) and (first_child.type == token.STRING) and (child.children[1].type == token.NEWLINE)):
                continue
            break
        elif (first_child.type == syms.import_from):
            module_name = first_child.children[1]
            if ((not isinstance(module_name, Leaf)) or (module_name.value != '__future__')):
                break
            imports |= set(get_imports_from_children(first_child.children[3:]))
        else:
            break
    return imports

@lru_cache()
def get_gitignore(root):
    ' Return a PathSpec matching gitignore content if present.'
    gitignore = (root / '.gitignore')
    lines: List[str] = []
    if gitignore.is_file():
        with gitignore.open() as gf:
            lines = gf.readlines()
    return PathSpec.from_lines('gitwildmatch', lines)

def normalize_path_maybe_ignore(path, root, report):
    'Normalize `path`. May return `None` if `path` was ignored.\n\n    `report` is where "path ignored" output goes.\n    '
    try:
        abspath = (path if path.is_absolute() else (Path.cwd() / path))
        normalized_path = abspath.resolve().relative_to(root).as_posix()
    except OSError as e:
        report.path_ignored(path, f'cannot be read because {e}')
        return None
    except ValueError:
        if path.is_symlink():
            report.path_ignored(path, f'is a symbolic link that points outside {root}')
            return None
        raise
    return normalized_path

def gen_python_files(paths, root, include, exclude, force_exclude, report, gitignore):
    'Generate all files under `path` whose paths are not excluded by the\n    `exclude_regex` or `force_exclude` regexes, but are included by the `include` regex.\n\n    Symbolic links pointing outside of the `root` directory are ignored.\n\n    `report` is where output about exclusions goes.\n    '
    assert root.is_absolute(), f'INTERNAL ERROR: `root` must be absolute but is {root}'
    for child in paths:
        normalized_path = normalize_path_maybe_ignore(child, root, report)
        if (normalized_path is None):
            continue
        if gitignore.match_file(normalized_path):
            report.path_ignored(child, 'matches the .gitignore file content')
            continue
        normalized_path = ('/' + normalized_path)
        if child.is_dir():
            normalized_path += '/'
        exclude_match = (exclude.search(normalized_path) if exclude else None)
        if (exclude_match and exclude_match.group(0)):
            report.path_ignored(child, 'matches the --exclude regular expression')
            continue
        force_exclude_match = (force_exclude.search(normalized_path) if force_exclude else None)
        if (force_exclude_match and force_exclude_match.group(0)):
            report.path_ignored(child, 'matches the --force-exclude regular expression')
            continue
        if child.is_dir():
            (yield from gen_python_files(child.iterdir(), root, include, exclude, force_exclude, report, gitignore))
        elif child.is_file():
            include_match = (include.search(normalized_path) if include else True)
            if include_match:
                (yield child)

@lru_cache()
def find_project_root(srcs):
    "Return a directory containing .git, .hg, or pyproject.toml.\n\n    That directory will be a common parent of all files and directories\n    passed in `srcs`.\n\n    If no directory in the tree contains a marker that would specify it's the\n    project root, the root of the file system is returned.\n    "
    if (not srcs):
        return Path('/').resolve()
    path_srcs = [Path(Path.cwd(), src).resolve() for src in srcs]
    src_parents = [(list(path.parents) + ([path] if path.is_dir() else [])) for path in path_srcs]
    common_base = max(set.intersection(*(set(parents) for parents in src_parents)), key=(lambda path: path.parts))
    for directory in (common_base, *common_base.parents):
        if (directory / '.git').exists():
            return directory
        if (directory / '.hg').is_dir():
            return directory
        if (directory / 'pyproject.toml').is_file():
            return directory
    return directory

@dataclass
class Report():
    'Provides a reformatting counter. Can be rendered with `str(report)`.'
    check = False
    diff = False
    quiet = False
    verbose = False
    change_count = 0
    same_count = 0
    failure_count = 0

    def done(self, src, changed):
        'Increment the counter for successful reformatting. Write out a message.'
        if (changed is Changed.YES):
            reformatted = ('would reformat' if (self.check or self.diff) else 'reformatted')
            if (self.verbose or (not self.quiet)):
                out(f'{reformatted} {src}')
            self.change_count += 1
        else:
            if self.verbose:
                if (changed is Changed.NO):
                    msg = f'{src} already well formatted, good job.'
                else:
                    msg = f"{src} wasn't modified on disk since last run."
                out(msg, bold=False)
            self.same_count += 1

    def failed(self, src, message):
        'Increment the counter for failed reformatting. Write out a message.'
        err(f'error: cannot format {src}: {message}')
        self.failure_count += 1

    def path_ignored(self, path, message):
        if self.verbose:
            out(f'{path} ignored: {message}', bold=False)

    @property
    def return_code(self):
        'Return the exit code that the app should use.\n\n        This considers the current state of changed files and failures:\n        - if there were any failures, return 123;\n        - if any files were changed and --check is being used, return 1;\n        - otherwise return 0.\n        '
        if self.failure_count:
            return 123
        elif (self.change_count and self.check):
            return 1
        return 0

    def __str__(self):
        'Render a color report of the current state.\n\n        Use `click.unstyle` to remove colors.\n        '
        if (self.check or self.diff):
            reformatted = 'would be reformatted'
            unchanged = 'would be left unchanged'
            failed = 'would fail to reformat'
        else:
            reformatted = 'reformatted'
            unchanged = 'left unchanged'
            failed = 'failed to reformat'
        report = []
        if self.change_count:
            s = ('s' if (self.change_count > 1) else '')
            report.append(click.style(f'{self.change_count} file{s} {reformatted}', bold=True))
        if self.same_count:
            s = ('s' if (self.same_count > 1) else '')
            report.append(f'{self.same_count} file{s} {unchanged}')
        if self.failure_count:
            s = ('s' if (self.failure_count > 1) else '')
            report.append(click.style(f'{self.failure_count} file{s} {failed}', fg='red'))
        return (', '.join(report) + '.')

def parse_ast(src):
    filename = '<unknown>'
    if (sys.version_info >= (3, 8)):
        for minor_version in range(sys.version_info[1], 4, (- 1)):
            try:
                return ast.parse(src, filename, feature_version=(3, minor_version))
            except SyntaxError:
                continue
    else:
        for feature_version in (7, 6):
            try:
                return ast3.parse(src, filename, feature_version=feature_version)
            except SyntaxError:
                continue
    return ast27.parse(src)

def _fixup_ast_constants(node):
    'Map ast nodes deprecated in 3.8 to Constant.'
    if isinstance(node, (ast.Str, ast3.Str, ast27.Str, ast.Bytes, ast3.Bytes)):
        return ast.Constant(value=node.s)
    if isinstance(node, (ast.Num, ast3.Num, ast27.Num)):
        return ast.Constant(value=node.n)
    if isinstance(node, (ast.NameConstant, ast3.NameConstant)):
        return ast.Constant(value=node.value)
    return node

def _stringify_ast(node, depth=0):
    'Simple visitor generating strings to compare ASTs by content.'
    node = _fixup_ast_constants(node)
    (yield f"{('  ' * depth)}{node.__class__.__name__}(")
    for field in sorted(node._fields):
        type_ignore_classes = (ast3.TypeIgnore, ast27.TypeIgnore)
        if (sys.version_info >= (3, 8)):
            type_ignore_classes += (ast.TypeIgnore,)
        if isinstance(node, type_ignore_classes):
            break
        try:
            value = getattr(node, field)
        except AttributeError:
            continue
        (yield f"{('  ' * (depth + 1))}{field}=")
        if isinstance(value, list):
            for item in value:
                if ((field == 'targets') and isinstance(node, (ast.Delete, ast3.Delete, ast27.Delete)) and isinstance(item, (ast.Tuple, ast3.Tuple, ast27.Tuple))):
                    for item in item.elts:
                        (yield from _stringify_ast(item, (depth + 2)))
                elif isinstance(item, (ast.AST, ast3.AST, ast27.AST)):
                    (yield from _stringify_ast(item, (depth + 2)))
        elif isinstance(value, (ast.AST, ast3.AST, ast27.AST)):
            (yield from _stringify_ast(value, (depth + 2)))
        else:
            if (isinstance(node, ast.Constant) and (field == 'value') and isinstance(value, str)):
                normalized = re.sub(' *\\n[ \\t]*', '\n', value).strip()
            else:
                normalized = value
            (yield f"{('  ' * (depth + 2))}{normalized!r},  # {value.__class__.__name__}")
    (yield f"{('  ' * depth)})  # /{node.__class__.__name__}")

def assert_equivalent(src, dst):
    "Raise AssertionError if `src` and `dst` aren't equivalent."
    try:
        src_ast = parse_ast(src)
    except Exception as exc:
        raise AssertionError(f'cannot use --safe with this file; failed to parse source file.  AST error message: {exc}')
    try:
        dst_ast = parse_ast(dst)
    except Exception as exc:
        log = dump_to_file(''.join(traceback.format_tb(exc.__traceback__)), dst)
        raise AssertionError(f'INTERNAL ERROR: Black produced invalid code: {exc}. Please report a bug on https://github.com/psf/black/issues.  This invalid output might be helpful: {log}') from None
    src_ast_str = '\n'.join(_stringify_ast(src_ast))
    dst_ast_str = '\n'.join(_stringify_ast(dst_ast))
    if (src_ast_str != dst_ast_str):
        log = dump_to_file(diff(src_ast_str, dst_ast_str, 'src', 'dst'))
        raise AssertionError(f'INTERNAL ERROR: Black produced code that is not equivalent to the source.  Please report a bug on https://github.com/psf/black/issues.  This diff might be helpful: {log}') from None

def assert_stable(src, dst, mode):
    'Raise AssertionError if `dst` reformats differently the second time.'
    newdst = format_str(dst, mode=mode)
    if (dst != newdst):
        log = dump_to_file(str(mode), diff(src, dst, 'source', 'first pass'), diff(dst, newdst, 'first pass', 'second pass'))
        raise AssertionError(f'INTERNAL ERROR: Black produced different code on the second pass of the formatter.  Please report a bug on https://github.com/psf/black/issues.  This diff might be helpful: {log}') from None

@mypyc_attr(patchable=True)
def dump_to_file(*output):
    'Dump `output` to a temporary file. Return path to the file.'
    with tempfile.NamedTemporaryFile(mode='w', prefix='blk_', suffix='.log', delete=False, encoding='utf8') as f:
        for lines in output:
            f.write(lines)
            if (lines and (lines[(- 1)] != '\n')):
                f.write('\n')
    return f.name

@contextmanager
def nullcontext():
    'Return an empty context manager.\n\n    To be used like `nullcontext` in Python 3.7.\n    '
    (yield)

def diff(a, b, a_name, b_name):
    'Return a unified diff string between strings `a` and `b`.'
    import difflib
    a_lines = [(line + '\n') for line in a.splitlines()]
    b_lines = [(line + '\n') for line in b.splitlines()]
    return ''.join(difflib.unified_diff(a_lines, b_lines, fromfile=a_name, tofile=b_name, n=5))

def cancel(tasks):
    'asyncio signal handler that cancels all `tasks` and reports to stderr.'
    err('Aborted!')
    for task in tasks:
        task.cancel()

def shutdown(loop):
    'Cancel all pending tasks on `loop`, wait for them, and close the loop.'
    try:
        if (sys.version_info[:2] >= (3, 7)):
            all_tasks = asyncio.all_tasks
        else:
            all_tasks = asyncio.Task.all_tasks
        to_cancel = [task for task in all_tasks(loop) if (not task.done())]
        if (not to_cancel):
            return
        for task in to_cancel:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*to_cancel, loop=loop, return_exceptions=True))
    finally:
        cf_logger = logging.getLogger('concurrent.futures')
        cf_logger.setLevel(logging.CRITICAL)
        loop.close()

def sub_twice(regex, replacement, original):
    'Replace `regex` with `replacement` twice on `original`.\n\n    This is used by string normalization to perform replaces on\n    overlapping matches.\n    '
    return regex.sub(replacement, regex.sub(replacement, original))

def re_compile_maybe_verbose(regex):
    'Compile a regular expression string in `regex`.\n\n    If it contains newlines, use verbose mode.\n    '
    if ('\n' in regex):
        regex = ('(?x)' + regex)
    compiled: Pattern[str] = re.compile(regex)
    return compiled

def enumerate_reversed(sequence):
    'Like `reversed(enumerate(sequence))` if that were possible.'
    index = (len(sequence) - 1)
    for element in reversed(sequence):
        (yield (index, element))
        index -= 1

def enumerate_with_length(line, reversed=False):
    'Return an enumeration of leaves with their length.\n\n    Stops prematurely on multiline strings and standalone comments.\n    '
    op = cast(Callable[([Sequence[Leaf]], Iterator[Tuple[(Index, Leaf)]])], (enumerate_reversed if reversed else enumerate))
    for (index, leaf) in op(line.leaves):
        length = (len(leaf.prefix) + len(leaf.value))
        if ('\n' in leaf.value):
            return
        for comment in line.comments_after(leaf):
            length += len(comment.value)
        (yield (index, leaf, length))

def is_line_short_enough(line, *, line_length, line_str=''):
    'Return True if `line` is no longer than `line_length`.\n\n    Uses the provided `line_str` rendering, if any, otherwise computes a new one.\n    '
    if (not line_str):
        line_str = line_to_string(line)
    return ((len(line_str) <= line_length) and ('\n' not in line_str) and (not line.contains_standalone_comments()))

def can_be_split(line):
    'Return False if the line cannot be split *for sure*.\n\n    This is not an exhaustive search but a cheap heuristic that we can use to\n    avoid some unfortunate formattings (mostly around wrapping unsplittable code\n    in unnecessary parentheses).\n    '
    leaves = line.leaves
    if (len(leaves) < 2):
        return False
    if ((leaves[0].type == token.STRING) and (leaves[1].type == token.DOT)):
        call_count = 0
        dot_count = 0
        next = leaves[(- 1)]
        for leaf in leaves[(- 2)::(- 1)]:
            if (leaf.type in OPENING_BRACKETS):
                if (next.type not in CLOSING_BRACKETS):
                    return False
                call_count += 1
            elif (leaf.type == token.DOT):
                dot_count += 1
            elif (leaf.type == token.NAME):
                if (not ((next.type == token.DOT) or (next.type in OPENING_BRACKETS))):
                    return False
            elif (leaf.type not in CLOSING_BRACKETS):
                return False
            if ((dot_count > 1) and (call_count > 1)):
                return False
    return True

def can_omit_invisible_parens(line, line_length, omit_on_explode=()):
    'Does `line` have a shape safe to reformat without optional parens around it?\n\n    Returns True for only a subset of potentially nice looking formattings but\n    the point is to not return false positives that end up producing lines that\n    are too long.\n    '
    bt = line.bracket_tracker
    if (not bt.delimiters):
        return True
    max_priority = bt.max_delimiter_priority()
    if (bt.delimiter_count_with_priority(max_priority) > 1):
        return False
    if (max_priority == DOT_PRIORITY):
        return True
    assert (len(line.leaves) >= 2), 'Stranded delimiter'
    first = line.leaves[0]
    second = line.leaves[1]
    if ((first.type in OPENING_BRACKETS) and (second.type not in CLOSING_BRACKETS)):
        if _can_omit_opening_paren(line, first=first, line_length=line_length):
            return True
    penultimate = line.leaves[(- 2)]
    last = line.leaves[(- 1)]
    if line.should_explode:
        try:
            (penultimate, last) = last_two_except(line.leaves, omit=omit_on_explode)
        except LookupError:
            return False
    if ((last.type == token.RPAR) or (last.type == token.RBRACE) or ((last.type == token.RSQB) and last.parent and (last.parent.type != syms.trailer))):
        if (penultimate.type in OPENING_BRACKETS):
            return False
        if is_multiline_string(first):
            return True
        if (line.should_explode and (penultimate.type == token.COMMA)):
            return True
        if _can_omit_closing_paren(line, last=last, line_length=line_length):
            return True
    return False

def _can_omit_opening_paren(line, *, first, line_length):
    'See `can_omit_invisible_parens`.'
    remainder = False
    length = (4 * line.depth)
    _index = (- 1)
    for (_index, leaf, leaf_length) in enumerate_with_length(line):
        if ((leaf.type in CLOSING_BRACKETS) and (leaf.opening_bracket is first)):
            remainder = True
        if remainder:
            length += leaf_length
            if (length > line_length):
                break
            if (leaf.type in OPENING_BRACKETS):
                remainder = False
    else:
        if (len(line.leaves) == (_index + 1)):
            return True
    return False

def _can_omit_closing_paren(line, *, last, line_length):
    'See `can_omit_invisible_parens`.'
    length = (4 * line.depth)
    seen_other_brackets = False
    for (_index, leaf, leaf_length) in enumerate_with_length(line):
        length += leaf_length
        if (leaf is last.opening_bracket):
            if (seen_other_brackets or (length <= line_length)):
                return True
        elif (leaf.type in OPENING_BRACKETS):
            seen_other_brackets = True
    return False

def last_two_except(leaves, omit):
    'Return (penultimate, last) leaves skipping brackets in `omit` and contents.'
    stop_after = None
    last = None
    for leaf in reversed(leaves):
        if stop_after:
            if (leaf is stop_after):
                stop_after = None
            continue
        if last:
            return (leaf, last)
        if (id(leaf) in omit):
            stop_after = leaf.opening_bracket
        else:
            last = leaf
    else:
        raise LookupError('Last two leaves were also skipped')

def run_transformer(line, transform, mode, features, *, line_str=''):
    if (not line_str):
        line_str = line_to_string(line)
    result: List[Line] = []
    for transformed_line in transform(line, features):
        if (str(transformed_line).strip('\n') == line_str):
            raise CannotTransform('Line transformer returned an unchanged result')
        result.extend(transform_line(transformed_line, mode=mode, features=features))
    if (not ((transform.__name__ == 'rhs') and line.bracket_tracker.invisible and (not any((bracket.value for bracket in line.bracket_tracker.invisible))) and (not line.contains_multiline_strings()) and (not result[0].contains_uncollapsable_type_comments()) and (not result[0].contains_unsplittable_type_ignore()) and (not is_line_short_enough(result[0], line_length=mode.line_length)))):
        return result
    line_copy = line.clone()
    append_leaves(line_copy, line, line.leaves)
    features_fop = (set(features) | {Feature.FORCE_OPTIONAL_PARENTHESES})
    second_opinion = run_transformer(line_copy, transform, mode, features_fop, line_str=line_str)
    if all((is_line_short_enough(ln, line_length=mode.line_length) for ln in second_opinion)):
        result = second_opinion
    return result

def get_cache_file(mode):
    return (CACHE_DIR / f'cache.{mode.get_cache_key()}.pickle')

def read_cache(mode):
    'Read the cache if it exists and is well formed.\n\n    If it is not well formed, the call to write_cache later should resolve the issue.\n    '
    cache_file = get_cache_file(mode)
    if (not cache_file.exists()):
        return {}
    with cache_file.open('rb') as fobj:
        try:
            cache: Cache = pickle.load(fobj)
        except (pickle.UnpicklingError, ValueError):
            return {}
    return cache

def get_cache_info(path):
    'Return the information used to check if a file is already formatted or not.'
    stat = path.stat()
    return (stat.st_mtime, stat.st_size)

def filter_cached(cache, sources):
    'Split an iterable of paths in `sources` into two sets.\n\n    The first contains paths of files that modified on disk or are not in the\n    cache. The other contains paths to non-modified files.\n    '
    (todo, done) = (set(), set())
    for src in sources:
        src = src.resolve()
        if (cache.get(src) != get_cache_info(src)):
            todo.add(src)
        else:
            done.add(src)
    return (todo, done)

def write_cache(cache, sources, mode):
    'Update the cache file.'
    cache_file = get_cache_file(mode)
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        new_cache = {**cache, **{src.resolve(): get_cache_info(src) for src in sources}}
        with tempfile.NamedTemporaryFile(dir=str(cache_file.parent), delete=False) as f:
            pickle.dump(new_cache, f, protocol=4)
        os.replace(f.name, cache_file)
    except OSError:
        pass

def patch_click():
    "Make Click not crash.\n\n    On certain misconfigured environments, Python 3 selects the ASCII encoding as the\n    default which restricts paths that it can access during the lifetime of the\n    application.  Click refuses to work in this scenario by raising a RuntimeError.\n\n    In case of Black the likelihood that non-ASCII characters are going to be used in\n    file paths is minimal since it's Python source code.  Moreover, this crash was\n    spurious on Python 3.7 thanks to PEP 538 and PEP 540.\n    "
    try:
        from click import core
        from click import _unicodefun
    except ModuleNotFoundError:
        return
    for module in (core, _unicodefun):
        if hasattr(module, '_verify_python3_env'):
            module._verify_python3_env = (lambda : None)

def patched_main():
    freeze_support()
    patch_click()
    main()

def is_docstring(leaf):
    if (not is_multiline_string(leaf)):
        return False
    if prev_siblings_are(leaf.parent, [None, token.NEWLINE, token.INDENT, syms.simple_stmt]):
        return True
    if prev_siblings_are(leaf.parent, [syms.parameters, token.COLON, syms.simple_stmt]):
        return True
    return False

def lines_with_leading_tabs_expanded(s):
    '\n    Splits string into lines and expands only leading tabs (following the normal\n    Python rules)\n    '
    lines = []
    for line in s.splitlines():
        match = re.match('\\s*\\t+\\s*(\\S)', line)
        if match:
            first_non_whitespace_idx = match.start(1)
            lines.append((line[:first_non_whitespace_idx].expandtabs() + line[first_non_whitespace_idx:]))
        else:
            lines.append(line)
    return lines

def fix_docstring(docstring, prefix):
    if (not docstring):
        return ''
    lines = lines_with_leading_tabs_expanded(docstring)
    indent = sys.maxsize
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, (len(line) - len(stripped)))
    trimmed = [lines[0].strip()]
    if (indent < sys.maxsize):
        last_line_idx = (len(lines) - 2)
        for (i, line) in enumerate(lines[1:]):
            stripped_line = line[indent:].rstrip()
            if (stripped_line or (i == last_line_idx)):
                trimmed.append((prefix + stripped_line))
            else:
                trimmed.append('')
    return '\n'.join(trimmed)
if (__name__ == '__main__'):
    patched_main()
