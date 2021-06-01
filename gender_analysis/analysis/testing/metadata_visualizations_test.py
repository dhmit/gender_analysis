# pylint: disable=invalid-name, missing-function-docstring, missing-class-docstring

import shutil
import filecmp
from pathlib import Path
from gender_analysis.text.common import create_path_object_and_directories
from gender_analysis.text.corpus import Corpus
from gender_analysis.testing import common


from ..metadata_visualizations import (
    plot_pubyears,
    plot_pubcountries,
    plot_gender_breakdown,
    plot_metadata_pie,
    create_corpus_summary_visualizations
)

OUTPUT_DIRECTORY_PATH = common.BASE_PATH / "testing" / "test_files" / "visualizations_test_dir"


class TestMetadataVisualizations:
    def test_create_path_object_and_directories(self):
        dir_path = OUTPUT_DIRECTORY_PATH / "additional_folder"
        input_path = dir_path / "random.jpg"
        output_path = create_path_object_and_directories(input_path)

        assert input_path == output_path
        assert Path.is_dir(dir_path)

        # clean up created folders
        shutil.rmtree(dir_path)

    def test_plot_pubyears_different_file_constructions(self):
        c = Corpus(
            common.TEST_CORPUS_PATH,
            csv_path=common.LARGE_TEST_CORPUS_CSV,
            name='test_corpus',
        )

        default_save_name = 'date_of_pub_for_' + c.name.replace(' ', '_') + '.png'
        test_file_1_name = "testing_file1.png"

        default_save_path = OUTPUT_DIRECTORY_PATH / default_save_name
        test_file_save_path = OUTPUT_DIRECTORY_PATH / test_file_1_name

        test_file_paths = []

        plot_pubyears(c, OUTPUT_DIRECTORY_PATH)
        assert Path.is_file(default_save_path)
        test_file_paths.append(default_save_path)

        plot_pubyears(c, OUTPUT_DIRECTORY_PATH, "testing file1")
        assert Path.is_file(test_file_save_path)
        test_file_paths.append(test_file_save_path)

        for file_created1 in test_file_paths:
            for file_created2 in test_file_paths:
                assert filecmp.cmp(file_created1, file_created2)

        for file_created in test_file_paths:
            Path.unlink(file_created)

    def test_plot_pubcountries_different_file_constructions(self):
        c = Corpus(
            common.TEST_CORPUS_PATH,
            csv_path=common.LARGE_TEST_CORPUS_CSV,
            name='test_corpus',
        )

        default_save_name = 'country_of_pub_for_' + c.name.replace(' ', '_') + '.png'
        test_file_1_name = "testing_file1.png"

        default_save_path = OUTPUT_DIRECTORY_PATH / default_save_name
        test_file_save_path = OUTPUT_DIRECTORY_PATH / test_file_1_name

        test_file_paths = []

        plot_pubcountries(c, OUTPUT_DIRECTORY_PATH)
        assert Path.is_file(default_save_path)
        test_file_paths.append(default_save_path)

        plot_pubcountries(c, OUTPUT_DIRECTORY_PATH, "testing file1")
        assert Path.is_file(test_file_save_path)
        test_file_paths.append(test_file_save_path)

        for file_created1 in test_file_paths:
            for file_created2 in test_file_paths:
                assert filecmp.cmp(file_created1, file_created2)

        for file_created in test_file_paths:
            Path.unlink(file_created)

    def test_plot_gender_breakdown_different_file_constructions(self):
        corpus = Corpus(
            common.TEST_CORPUS_PATH,
            csv_path=common.LARGE_TEST_CORPUS_CSV,
            name='test_corpus',
        )

        default_save_name = 'gender_breakdown_for_' + corpus.name.replace(' ', '_') + '.png'
        test_file_1_name = "testing_file1.png"

        default_save_path = OUTPUT_DIRECTORY_PATH / default_save_name
        test_file_save_path = OUTPUT_DIRECTORY_PATH / test_file_1_name

        test_file_paths = []

        plot_gender_breakdown(corpus, OUTPUT_DIRECTORY_PATH)
        assert Path.is_file(default_save_path)
        test_file_paths.append(default_save_path)

        plot_gender_breakdown(corpus, OUTPUT_DIRECTORY_PATH, "testing file1")
        assert Path.is_file(test_file_save_path)
        test_file_paths.append(test_file_save_path)

        for file_created1 in test_file_paths:
            for file_created2 in test_file_paths:
                assert filecmp.cmp(file_created1, file_created2)

        for file_created in test_file_paths:
            Path.unlink(file_created)

    def test_plot_metadata_pie_different_file_constructions(self):
        corpus = Corpus(
            common.TEST_CORPUS_PATH,
            csv_path=common.LARGE_TEST_CORPUS_CSV,
            name='test_corpus',
        )

        save_name = 'percentage_acquired_metadata_for_' + corpus.name.replace(' ', '_') + '.png'
        test_file_1_name = "testing_file1.png"

        save_path = OUTPUT_DIRECTORY_PATH / save_name
        test_file_save_path = OUTPUT_DIRECTORY_PATH / test_file_1_name

        test_file_paths = []

        plot_metadata_pie(corpus, OUTPUT_DIRECTORY_PATH)
        assert Path.is_file(save_path)
        test_file_paths.append(save_path)

        plot_metadata_pie(corpus, OUTPUT_DIRECTORY_PATH, "testing file1")
        assert Path.is_file(test_file_save_path)
        test_file_paths.append(test_file_save_path)

        for file_created1 in test_file_paths:
            for file_created2 in test_file_paths:
                assert filecmp.cmp(file_created1, file_created2)

        for file_created in test_file_paths:
            Path.unlink(file_created)

    def test_create_all_visualizations_but_with_no_corpus_name(self):
        corpus = Corpus(
            common.TEST_CORPUS_PATH,
            csv_path=common.LARGE_TEST_CORPUS_CSV
        )

        default_gender_breakdown = 'gender_breakdown_for_corpus.png'
        default_metadata_pie = 'percentage_acquired_metadata_for_corpus.png'
        default_country_pub = 'country_of_pub_for_corpus.png'
        default_pub_date = 'date_of_pub_for_corpus.png'

        create_corpus_summary_visualizations(corpus, OUTPUT_DIRECTORY_PATH)
        assert Path.is_file(OUTPUT_DIRECTORY_PATH / default_gender_breakdown)
        assert Path.is_file(OUTPUT_DIRECTORY_PATH / default_pub_date)
        assert Path.is_file(OUTPUT_DIRECTORY_PATH / default_country_pub)
        assert Path.is_file(OUTPUT_DIRECTORY_PATH / default_metadata_pie)
