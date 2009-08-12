#!/usr/bin/env python

import unittest

import test_specex

def main():
    suite = unittest.TestSuite()
    suite.addTest(test_specex.suite())
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == '__main__':
    main()

# vim: et sw=4 sts=4
