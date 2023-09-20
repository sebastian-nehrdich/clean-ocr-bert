# clean-ocr-bert
This is a simple script that cleans up OCRed files for the MITRA project.  
There are two training scripts for the classifiers:  

train_classifier_next_sentence.py: Detect if two sentences are in consecutive order  

train_classifier_footnote_detection.py: Detect if a sentence belongs to a footnote or to the main translation  

clean_ocr.py then combines these two classifiers in order to apply them to an unseen text file.  

