#!/bin/bash
serverIp=159.89.154.87
git clone --recurse-submodules https://github.com/snsnlou2/mypy.git
pip install -U ./mypy
rm -rf ./mypy
python3 typecheck.py
eval "$(ssh-agent -s)"
chmod 600 root_key
ssh-keyscan $serverIp >> ~/.ssh/known_hosts
ssh-add root_key
zip -r mypy_test_cache.zip mypy_test_cache/
yes | scp -i root_key ./mypy_test_cache.zip root@$serverIp:~/cache/psf---black--unannotated.zip
yes | scp -i root_key ./mypy_test_report.txt root@$serverIp:~/report/psf---black--unannotated.txt
