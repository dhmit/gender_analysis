import shutil
from gender_analysis.analysis.metadata_visualizations import *
from gender_analysis.corpus import Corpus
from gender_analysis.testing import common
import filecmp

OUTPUT_DIRECTORY_PATH = Path(os.getcwd()).joinpath(Path("gender_analysis")) \
    .joinpath(Path("testing")).joinpath(Path("test_files")) \
    .joinpath(Path("visualizations_test_directory"))


class TestMetadataVisualizations:

    def test_generate_filepath_for_visualizations_create_directory(self):
        if os.path.isdir(OUTPUT_DIRECTORY_PATH):
            shutil.rmtree(OUTPUT_DIRECTORY_PATH)

        assert generate_file_path_for_visualizations(OUTPUT_DIRECTORY_PATH) == OUTPUT_DIRECTORY_PATH
        assert generate_file_path_for_visualizations(
            str(OUTPUT_DIRECTORY_PATH)) == OUTPUT_DIRECTORY_PATH
        assert os.path.isdir(OUTPUT_DIRECTORY_PATH)

        # Test a bad directory creation
        assert generate_file_path_for_visualizations(
            OUTPUT_DIRECTORY_PATH.joinpath("additional_folder").joinpath(
                "another_folder")) == DEFAULT_VISUALIZATION_OUTPUT_DIR

    def test_plot_pubyears_different_file_constructions(self):
        c = Corpus(
            common.LARGE_TEST_CORPUS_PATH,
            csv_path=common.LARGE_TEST_CORPUS_CSV,
            name='test_corpus',
        )

        default_save_name = 'date_of_pub_for_' + c.name.replace(' ', '_') + '.png'
        test_file_1_name = "testing_file1.png"

        test_file_paths = []

        plot_pubyears(c, OUTPUT_DIRECTORY_PATH)
        assert os.path.isfile(OUTPUT_DIRECTORY_PATH.joinpath(default_save_name))
        test_file_paths.append(OUTPUT_DIRECTORY_PATH.joinpath(default_save_name))

        plot_pubyears(c, OUTPUT_DIRECTORY_PATH, "testing file1")
        assert os.path.isfile(OUTPUT_DIRECTORY_PATH.joinpath(test_file_1_name))
        test_file_paths.append(OUTPUT_DIRECTORY_PATH.joinpath(test_file_1_name))

        for file_created1 in test_file_paths:
            for file_created2 in test_file_paths:
                assert filecmp.cmp(file_created1, file_created2)

        for file_created in test_file_paths:
            os.remove(file_created)

    def test_plot_pubcountries_different_file_constructions(self):
        c = Corpus(
            common.LARGE_TEST_CORPUS_PATH,
            csv_path=common.LARGE_TEST_CORPUS_CSV,
            name='test_corpus',
        )

        default_save_name = 'country_of_pub_for_' + c.name.replace(' ', '_') + '.png'
        test_file_1_name = "testing_file1.png"

        test_file_paths = []

        plot_pubcountries(c, OUTPUT_DIRECTORY_PATH)
        assert os.path.isfile(OUTPUT_DIRECTORY_PATH.joinpath(default_save_name))
        test_file_paths.append(OUTPUT_DIRECTORY_PATH.joinpath(default_save_name))

        plot_pubcountries(c, OUTPUT_DIRECTORY_PATH, "testing file1")
        assert os.path.isfile(OUTPUT_DIRECTORY_PATH.joinpath(test_file_1_name))
        test_file_paths.append(OUTPUT_DIRECTORY_PATH.joinpath(test_file_1_name))

        for file_created1 in test_file_paths:
            for file_created2 in test_file_paths:
                assert filecmp.cmp(file_created1, file_created2)

        for file_created in test_file_paths:
            os.remove(file_created)

    def test_plot_gender_breakdown_different_file_constructions(self):
        c = Corpus(
            common.LARGE_TEST_CORPUS_PATH,
            csv_path=common.LARGE_TEST_CORPUS_CSV,
            name='test_corpus',
        )

        default_save_name = 'gender_breakdown_for_' + c.name.replace(' ', '_') + '.png'
        test_file_1_name = "testing_file1.png"

        test_file_paths = []

        plot_gender_breakdown(c, OUTPUT_DIRECTORY_PATH)
        assert os.path.isfile(OUTPUT_DIRECTORY_PATH.joinpath(default_save_name))
        test_file_paths.append(OUTPUT_DIRECTORY_PATH.joinpath(default_save_name))

        plot_gender_breakdown(c, OUTPUT_DIRECTORY_PATH, "testing file1")
        assert os.path.isfile(OUTPUT_DIRECTORY_PATH.joinpath(test_file_1_name))
        test_file_paths.append(OUTPUT_DIRECTORY_PATH.joinpath(test_file_1_name))

        for file_created1 in test_file_paths:
            for file_created2 in test_file_paths:
                assert filecmp.cmp(file_created1, file_created2)

        for file_created in test_file_paths:
            os.remove(file_created)

    def test_plot_metadata_pie_different_file_constructions(self):
        c = Corpus(
            common.LARGE_TEST_CORPUS_PATH,
            csv_path=common.LARGE_TEST_CORPUS_CSV,
            name='test_corpus',
        )

        default_save_name = 'percentage_acquired_metadata_for_' + c.name.replace(' ', '_') + '.png'
        test_file_1_name = "testing_file1.png"

        test_file_paths = []

        plot_metadata_pie(c, OUTPUT_DIRECTORY_PATH)
        assert os.path.isfile(OUTPUT_DIRECTORY_PATH.joinpath(default_save_name))
        test_file_paths.append(OUTPUT_DIRECTORY_PATH.joinpath(default_save_name))

        plot_metadata_pie(c, OUTPUT_DIRECTORY_PATH, "testing file1")
        assert os.path.isfile(OUTPUT_DIRECTORY_PATH.joinpath(test_file_1_name))
        test_file_paths.append(OUTPUT_DIRECTORY_PATH.joinpath(test_file_1_name))

        for file_created1 in test_file_paths:
            for file_created2 in test_file_paths:
                assert filecmp.cmp(file_created1, file_created2)

        for file_created in test_file_paths:
            os.remove(file_created)

    def test_create_all_visualizations_but_with_no_corpus_name(self):
        c = Corpus(
            common.LARGE_TEST_CORPUS_PATH,
            csv_path=common.LARGE_TEST_CORPUS_CSV
        )

        default_gender_breakdown = 'gender_breakdown_for_corpus.png'
        default_metadata_pie = 'percentage_acquired_metadata_for_corpus.png'
        default_country_pub = 'country_of_pub_for_corpus.png'
        default_pub_date = 'date_of_pub_for_corpus.png'

        create_corpus_summary_visualizations(c, OUTPUT_DIRECTORY_PATH)
        assert os.path.isfile(OUTPUT_DIRECTORY_PATH.joinpath(default_gender_breakdown))
        assert os.path.isfile(OUTPUT_DIRECTORY_PATH.joinpath(default_pub_date))
        assert os.path.isfile(OUTPUT_DIRECTORY_PATH.joinpath(default_country_pub))
        assert os.path.isfile(OUTPUT_DIRECTORY_PATH.joinpath(default_metadata_pie))
