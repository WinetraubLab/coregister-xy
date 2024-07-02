import unittest

def run_all_tests():
    loader = unittest.TestLoader()
    test_suite = loader.discover('.', pattern='test_*.py')
    test_runner = unittest.TextTestRunner()
    test_runner.run(test_suite)

if __name__ == "__main__":
    run_all_tests()
