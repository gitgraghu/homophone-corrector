# **Homophone Corrector**

## **Approach:**

The classification approach used involves determining context of a possibly incorrect word by using its surrounding words and their POS tags.
The features used for training are the two words to the left and their POS tags and two words to the right and their POS tags.
The NLTK Maximum Entropy classifier was used for training the model. Out of the varous Maxent algorithms MEGAM gave the best results in terms of performance and accuracy.
The model was trained using Brown Corpus, Abc Corpus and 100MB Wikipedia dumps.

## **Running the Code:**

[1.] Training Data Model

    - python homophonetrainer.py (will generate model file 'model.pickle')

[2.] Classifying Input Data:

    - python homophonecorrector.py < hw3.test.err.txt > hw3.output.txt

## **Third party software used:**

List of 3rd party software used

- [NLTK](http://www.nltk.org/)

- [NLTK Brown and ABC Corpus](http://www.nltk.org/data.html)

- [MEGAM](http://www.umiacs.umd.edu/~hal/megam/)

- [Wiki Dumps](http://en.wikipedia.org/wiki/Wikipedia%3aDatabase_download)