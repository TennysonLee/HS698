import unittest

from requester import url_to_csv, batch_url_to_csv, url_to_df


# unittest documentation
valid_csv_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
valid_csv_url_v2 = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
invalid_url = "http://www.example.com/junk"
invalid_csv_url = "http://stackoverflow.com/questions/1393324/in-python-given-a-url-to-a-text-file-what-is-the-simplest-way-to-read-the-cont"
invalid_csv_url_v2 = "https://archive.ics.uci.edu/ml/datasets/Adult"


class TestURLToCSV(unittest.TestCase):

    def test_type_error_v1(self): # Unit Test #1
        with self.assertRaises(TypeError):
            url_to_csv(invalid_csv_url)


    def test_type_error_v2(self): # Unit Test #1
        with self.assertRaises(TypeError):
            url_to_csv(invalid_csv_url_v2)


    def test_value_error_v2(self): # Unit Test #2
        with self.assertRaises(ValueError):
            url_to_csv(invalid_url)


class TestBatchURLToCSV(unittest.TestCase):

    def test_invalid_URLs_not_raising_errors(self): # Unit Test #3
        urls = [valid_csv_url, valid_csv_url_v2, invalid_url, invalid_csv_url, invalid_csv_url_v2]
        filenames = ["valid_csv_url", "valid_csv_url_v2", "invalid_url", "invalid_csv_url", "invalid_csv_url_v2"]
        res = batch_url_to_csv(urls, filenames)
        self.assertEqual(res, ["valid_csv_url.csv", "valid_csv_url_v2.csv"])

    def test_correct_number_of_files_created(self): # Unit Test #4
        urls = [valid_csv_url, valid_csv_url_v2, invalid_url, invalid_csv_url, invalid_csv_url_v2]
        filenames = ["valid_csv_url", "valid_csv_url_v2", "invalid_url", "invalid_csv_url", "invalid_csv_url_v2"]
        res = batch_url_to_csv(urls, filenames)
        self.assertEqual(len(res), 2)

