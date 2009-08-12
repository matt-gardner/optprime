coverage -x test/run_tests.py
coverage -b -i -d test/htmlcov
firefox test/htmlcov/index.html
