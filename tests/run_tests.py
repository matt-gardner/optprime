#!/usr/bin/env python

import unittest

import test_specex
import test_specmethod

def main():
    suite = unittest.TestSuite()
    suite.addTest(test_specex.suite())
    suite.addTest(test_specmethod.suite())
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == '__main__':
    main()

# vim: et sw=4 sts=4
