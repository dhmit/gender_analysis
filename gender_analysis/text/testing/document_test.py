# pylint: disable=invalid-name, missing-function-docstring

import csv
import pytest

from gender_analysis.text.document import Document
from gender_analysis.testing import common


class TestDocumentInitialization:
    """
    Tests that the `Document` class
    """

    def test_sample_novels_document_initialization(self):
        documents = []
        with open(common.LARGE_TEST_CORPUS_CSV, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            num_documents = 0
            for row in reader:
                row["filepath"] = common.TEST_CORPUS_PATH / row["filename"]
                documents.append(Document(row))
                num_documents += 1
            assert num_documents == len(documents)

    def test_document_initialization_wrong_parameters(self):
        with pytest.raises(TypeError):
            Document([])

    def test_document_initialization_missing_filepath_in_metadata(self):
        with pytest.raises(ValueError):
            Document({})

    def test_document_initialization_metadata_file_with_wrong_extension(self):
        with pytest.raises(ValueError):
            with open(common.SMALL_TEST_CORPUS_CSV, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    row["filename"] = row["filename"].replace(".txt", "")
                    Document(row)

    def test_document_initialization_disallowed_field_in_metadata(self):
        with pytest.raises(KeyError):
            with open(common.SMALL_TEST_CORPUS_CSV, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    row["members"] = "member"
                    Document(row)

    def test_document_initialization_incorrect_date(self):
        with pytest.raises(ValueError):
            with open(common.SMALL_TEST_CORPUS_CSV, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    row["date"] = "foobar"
                    Document(row)
