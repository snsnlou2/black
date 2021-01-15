
'This script runs mypy on all py files in the given directory.\n\nAuthor: Shining (Fred) Lou\n'
import os
import sys
from subprocess import Popen, PIPE
from typing import List, Tuple

def get_src_file_list(dir):
    ' Get all py files from the given directories\n    '
    res = []
    for (root, dirs, files) in os.walk(dir):
        dirs.sort()
        for file_name in sorted(files):
            if (file_name[(- 3):] != '.py'):
                continue
            full_path = os.path.join(root, file_name)
            if (full_path != './typecheck.py'):
                res.append(full_path)
    return res

def run_mypy(files=[]):
    ' Call mypy given list of files, packages, modules and arguments\n    Output: stdout, stderr\n    '
    args = ['mypy', '--follow-imports', 'silent', '--ignore-missing-imports', '--check-untyped-defs', '--sqlite-cache', '--show-error-codes']
    command = (args + files)
    print(command)
    proc = Popen(command, stdout=PIPE)
    output = proc.communicate()
    out = ''
    if output[0]:
        out = output[0].decode('utf-8')
    err = ''
    if output[1]:
        err = output[1].decode('utf-8')
    return (out, err)

def process_duplicates(files):
    file_name_set = set()
    duplicate_set = set()
    for file in files:
        file_split = file.split('/')
        file_name = file_split[(len(file_split) - 1)]
        if (file_name not in file_name_set):
            file_name_set.add(file_name)
        else:
            duplicate_set.add(file_name)
    base_file_list = []
    duplicate_file_list = []
    for file in files:
        file_split = file.split('/')
        file_name = file_split[(len(file_split) - 1)]
        if (file_name not in duplicate_set):
            base_file_list.append(file)
        else:
            duplicate_file_list.append(file)
    return (base_file_list, duplicate_file_list)

def main():
    src_files = get_src_file_list('.')
    (base_file_list, duplicate_file_list) = process_duplicates(src_files)
    python_version = '{}.{}'.format(sys.version_info.major, sys.version_info.minor)
    os.mkdir('mypy_test_cache')
    with open('mypy_test_report.txt', 'a') as file:
        (output, _) = run_mypy(base_file_list)
        file.write(output)
        os.rename('./.mypy_cache/{}/cache.db'.format(python_version), './mypy_test_cache/main_cache.db')
        i = 0
        for dup_file in duplicate_file_list:
            (individual_output, _) = run_mypy([dup_file])
            file.write(individual_output)
            os.rename('./.mypy_cache/{}/cache.db'.format(python_version), './mypy_test_cache/duplicate_cache({}).db'.format(i))
            i += 1
if (__name__ == '__main__'):
    main()
