from flask import Flask, request, render_template
#in gender_analysis
from gender_analysis import document, Corpus
from gender_analysis.common import *
from gender_analysis.testing.common import *
from gender_analysis.analysis.gender_adjective import *

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def subsub():
    error = None
    if request.method == "GET":
        return render_template("subsub_form.html")
    elif request.method == "POST":
        corpus_to_search_in = request.form["corpus"]
        operation = request.form["operation"]
        genders = request.form["genders"]
        number_to_return = request.form["number"]

        if corpus_to_search_in == "small":
            corpus = Corpus(TEST_CORPUS_PATH, csv_path=SMALL_TEST_CORPUS_CSV,
                            ignore_warnings=True)
        elif corpus_to_search_in == "large":
            corpus = Corpus(TEST_CORPUS_PATH, csv_path=LARGE_TEST_CORPUS_CSV,
                            ignore_warnings=True)

        if operation == "diff_adj":
            result_dict = {}
            adj_analysis = run_adj_analysis(corpus, gender_list=genders)
            for doc in adj_analysis:
                result_dict[doc] = difference_adjs(adj_analysis[doc], number_to_return)
            return render_template("diff_adj_template.html", results=result_dict)
