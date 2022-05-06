# sumreads

A project to summarize reviews from [goodreads](https://www.goodreads.com/).

## Run

At the present moment, the project can only be run locally and requires download of [prepared review data](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home).

After cloning the project, it might be necessary to install dependencies. There is a `requirements.txt` file; after installing requirements also install `spacy install en_core_web_md` and some `nltk` packages (they pop up during the use with a note of what command to run).

After that, download the review data for you relevant genres and ungz them to the `data` folder.

### Use

The most convenient way to use is the provided Jupyter Notebook. As of now, you have to manually set id of the book (first number in the goodreads book's url). Possible summarizations are all used in the notebook and you can tweak the constants to your liking.
