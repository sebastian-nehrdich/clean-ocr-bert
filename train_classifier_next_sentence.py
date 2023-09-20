from simpletransformers.classification import  ClassificationModel, ClassificationArgs
import pandas as pd
import logging
import sklearn

train_df = pd.read_csv("data-next-sentence/train.tsv", sep="\t", names=["text", "labels"])[:200000]
eval_df = pd.read_csv("data-next-sentence/val.tsv", sep="\t", names=["text", "labels"])
test_df = pd.read_csv("data-next-sentence/test.tsv", sep="\t", names=["text", "labels"])
# convert labels into binary values
print(train_df)


model_args = ClassificationArgs()
model_args.num_train_epochs = 10
model_args.learning_rate = 1e-5
model_args.train_batch_size = 16
model_args.eval_batch_size = 16
model_args.overwrite_output_dir = True
model_args.max_seq_length = 512
model_args.evaluate_during_training = True
model_args.evaluate_during_training_steps = 200
model_args.evaluate_during_training_verbose = True
model_args.save_model_every_epoch = False
model_args.use_multiprocessing = False
model_args.use_multiprocessing_for_evaluation = False
model_args.output_dir = "output-next-sentence/"


model = ClassificationModel('bert', 'bert-base-cased', args=model_args, use_cuda=True)

model.train_model(train_df, eval_df=eval_df)

result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score)