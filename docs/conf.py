
from pathlib import Path
import re
import string
from typing import Callable, Dict, List, Optional, Pattern, Tuple, Set
from dataclasses import dataclass
import logging
from pkg_resources import get_distribution
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
LOG = logging.getLogger(__name__)
CURRENT_DIR = Path(__file__).parent
README = ((CURRENT_DIR / '..') / 'README.md')
REFERENCE_DIR = (CURRENT_DIR / 'reference')
STATIC_DIR = (CURRENT_DIR / '_static')

@dataclass
class SrcRange():
    "Tracks which part of a file to get a section's content.\n\n    Data:\n        start_line: The line where the section starts (i.e. its sub-header) (inclusive).\n        end_line: The line where the section ends (usually next sub-header) (exclusive).\n    "

@dataclass
class DocSection():
    "Tracks information about a section of documentation.\n\n    Data:\n        name: The section's name. This will used to detect duplicate sections.\n        src: The filepath to get its contents.\n        processors: The processors to run before writing the section to CURRENT_DIR.\n        out_filename: The filename to use when writing the section to CURRENT_DIR.\n        src_range: The line range of SRC to gets its contents.\n    "
    src_range = SrcRange(0, 1000000)
    out_filename = ''
    processors = ()

    def get_out_filename(self):
        if (not self.out_filename):
            return (self.name + '.md')
        else:
            return self.out_filename

def make_pypi_svg(version):
    template: Path = ((CURRENT_DIR / '_static') / 'pypi_template.svg')
    target: Path = ((CURRENT_DIR / '_static') / 'pypi.svg')
    with open(str(template), 'r', encoding='utf8') as f:
        svg: str = string.Template(f.read()).substitute(version=version)
    with open(str(target), 'w', encoding='utf8') as f:
        f.write(svg)

def make_filename(line):
    non_letters: Pattern = re.compile('[^a-z]+')
    filename: str = line[3:].rstrip().lower()
    filename = non_letters.sub('_', filename)
    if filename.startswith('_'):
        filename = filename[1:]
    if filename.endswith('_'):
        filename = filename[:(- 1)]
    return (filename + '.md')

def get_contents(section):
    'Gets the contents for the DocSection.'
    contents: List[str] = []
    src: Path = section.src
    start_line: int = section.src_range.start_line
    end_line: int = section.src_range.end_line
    with open(src, 'r', encoding='utf-8') as f:
        for (lineno, line) in enumerate(f, start=1):
            if ((lineno >= start_line) and (lineno < end_line)):
                contents.append(line)
    result = ''.join(contents)
    if result.endswith('\n\n'):
        result = result[:(- 1)]
    if (not result.endswith('\n')):
        result = (result + '\n')
    return result

def get_sections_from_readme():
    'Gets the sections from README so they can be processed by process_sections.\n\n    It opens README and goes down line by line looking for sub-header lines which\n    denotes a section. Once it finds a sub-header line, it will create a DocSection\n    object with all of the information currently available. Then on every line, it will\n    track the ending line index of the section. And it repeats this for every sub-header\n    line it finds.\n    '
    sections: List[DocSection] = []
    section: Optional[DocSection] = None
    with open(README, 'r', encoding='utf-8') as f:
        for (lineno, line) in enumerate(f, start=1):
            if line.startswith('## '):
                filename = make_filename(line)
                section_name = filename[:(- 3)]
                section = DocSection(name=str(section_name), src=README, src_range=SrcRange(lineno, lineno), out_filename=filename, processors=(fix_headers,))
                sections.append(section)
            if (section is not None):
                section.src_range.end_line += 1
    return sections

def fix_headers(contents):
    'Fixes the headers of sections copied from README.\n\n    Removes one octothorpe (#) from all headers since the contents are no longer nested\n    in a root document (i.e. the README).\n    '
    lines: List[str] = contents.splitlines()
    fixed_contents: List[str] = []
    for line in lines:
        if line.startswith('##'):
            line = line[1:]
        fixed_contents.append((line + '\n'))
    return ''.join(fixed_contents)

def process_sections(custom_sections, readme_sections):
    'Reads, processes, and writes sections to CURRENT_DIR.\n\n    For each section, the contents will be fetched, processed by processors\n    required by the section, and written to CURRENT_DIR. If it encounters duplicate\n    sections (i.e. shares the same name attribute), it will skip processing the\n    duplicates.\n\n    It processes custom sections before the README generated sections so sections in the\n    README can be overwritten with custom options.\n    '
    processed_sections: Dict[(str, DocSection)] = {}
    modified_files: Set[Path] = set()
    sections: List[DocSection] = custom_sections
    sections.extend(readme_sections)
    for section in sections:
        if (section.name in processed_sections):
            LOG.warning(f"Skipping '{section.name}' from '{section.src}' as it is a duplicate of a custom section from '{processed_sections[section.name].src}'")
            continue
        LOG.info(f"Processing '{section.name}' from '{section.src}'")
        target_path: Path = (CURRENT_DIR / section.get_out_filename())
        if (target_path in modified_files):
            LOG.warning(f'{target_path} has been already written to, its contents will be OVERWRITTEN and notices will be duplicated')
        contents: str = get_contents(section)
        if (fix_headers in section.processors):
            contents = fix_headers(contents)
        with open(target_path, 'w', encoding='utf-8') as f:
            if ((section.src.suffix == '.md') and (section.src != target_path)):
                rel = section.src.resolve().relative_to(CURRENT_DIR.parent)
                f.write(f'''[//]: # "NOTE: THIS FILE WAS AUTOGENERATED FROM {rel}"

''')
            f.write(contents)
        processed_sections[section.name] = section
        modified_files.add(target_path)
project = 'Black'
copyright = '2020, Łukasz Langa and contributors to Black'
author = 'Łukasz Langa and contributors to Black'
release = get_distribution('black').version.split('+')[0]
version = release
for sp in 'abcfr':
    version = version.split(sp)[0]
custom_sections = [DocSection('the_black_code_style', (CURRENT_DIR / 'the_black_code_style.md')), DocSection('editor_integration', (CURRENT_DIR / 'editor_integration.md')), DocSection('blackd', (CURRENT_DIR / 'blackd.md')), DocSection('black_primer', (CURRENT_DIR / 'black_primer.md')), DocSection('contributing_to_black', ((CURRENT_DIR / '..') / 'CONTRIBUTING.md')), DocSection('change_log', ((CURRENT_DIR / '..') / 'CHANGES.md'))]
blocklisted_sections_from_readme = {'license', 'pragmatism', 'testimonials', 'used_by'}
make_pypi_svg(release)
readme_sections = get_sections_from_readme()
readme_sections = [x for x in readme_sections if (x.name not in blocklisted_sections_from_readme)]
process_sections(custom_sections, readme_sections)
needs_sphinx = '3.0'
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.intersphinx', 'sphinx.ext.napoleon', 'recommonmark']
needs_extensions = {'recommonmark': '0.5'}
templates_path = ['_templates']
source_suffix = ['.rst', '.md']
master_doc = 'index'
language = None
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = 'sphinx'
html_theme = 'alabaster'
html_sidebars = {'**': ['about.html', 'navigation.html', 'relations.html', 'sourcelink.html', 'searchbox.html']}
html_theme_options = {'show_related': False, 'description': '“Any color you like.”', 'github_button': True, 'github_user': 'psf', 'github_repo': 'black', 'github_type': 'star', 'show_powered_by': True, 'fixed_sidebar': True, 'logo': 'logo2.png', 'travis_button': True}
html_static_path = ['_static']
htmlhelp_basename = 'blackdoc'
latex_elements = {}
latex_documents = [(master_doc, 'black.tex', 'Documentation for Black', 'Łukasz Langa and contributors to Black', 'manual')]
man_pages = [(master_doc, 'black', 'Documentation for Black', [author], 1)]
texinfo_documents = [(master_doc, 'Black', 'Documentation for Black', author, 'Black', 'The uncompromising Python code formatter', 'Miscellaneous')]
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright
epub_exclude_files = ['search.html']
autodoc_member_order = 'bysource'
intersphinx_mapping = {'https://docs.python.org/3/': None}
