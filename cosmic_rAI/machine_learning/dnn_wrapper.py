import tensorflow as tf
import pandas as pd
import numpy as np

'''
Given a dataframe where each column is a feature, and each row is a value, return a dictionary:
{column: [values]}
'''
def dict_from_df(df):
    d = {}
    for column in df.columns:
        d[str(column)] = df[column].values
    return d

'''
Given a dataframe, replace all the zeros with a small number. Used so we can take the log later on.
'''
def no_zeros(df):
    return df.replace(0, 1e-8)

'''
Given a dataframe of the format created by data_prep.data_prep_pandas.event_df_from_matrices(),
return an array containing 1 and 0, where 0=PPlus and 1=Fe56Nucleus.
Namely, we're expecting a multiindexing scheme, where a top-level column called 'composition' contains
a column with values 'PPlus' and 'Fe56Nucleus'.
'''
def get_labels(df):
    return df.loc[:, ('composition', 0)].replace(['PPlus', 'Fe56Nucleus'], [0,1]).values

def get_flattened_labels(df):
    return df.loc[:, 'composition_0'].replace(['PPlus', 'Fe56Nucleus'], [0,1]).values

#Taken from https://www.tensorflow.org/get_started/datasets_quickstart
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(buffer_size=30000).repeat(count=None).batch(batch_size)

    # Build the Iterator, and return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()

def eval_input_fn(features, labels=None, batch_size=None):
    """An input function for evaluation or prediction"""
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert inputs to a tf.dataset object.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()

'''
Input:
train_data: a non-multiindexed dataframe containing each training sample and feature
train_labels: an array of numeric values of the same length as the number of rows in train_data. 
    Each number corresponds to a class.
test_data: a non-multiindexed dataframe containing each testing sample and feature
test_labels: an array of numeric values of the same length as the number of rows in test_data.
    Each number corresponds to a class.
'''
def one_run(train_data, train_labels, test_data, test_labels, steps):
    train_x = dict_from_df(no_zeros(train_data))
    train_y = train_labels
    test_x = dict_from_df(no_zeros(test_data))
    test_y = test_labels

#     feature_columns = []
    feature_columns = [tf.feature_column.numeric_column(key=str(column)) for column in train_x.keys()]
#     for column in train_x.keys():
#         feature_columns.append(tf.feature_column.numeric_column(key=str(column)))

    classifier = tf.estimator.DNNClassifier(
        feature_columns = feature_columns,
        hidden_units = [81, 40],
        n_classes = 2)

    classifier.train(
        input_fn=lambda:train_input_fn(train_x,train_y,100), steps=steps)

    eval_result = classifier.evaluate(
        input_fn=lambda:eval_input_fn(test_x, test_y, 100))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    return classifier

'''
df - the dataframe to split
features - the features to use in classification
steps - the number of steps to run 
'''
def split_and_run(df, features, steps):
    df = df.sample(frac=1)
    train_size = int(df.shape[0] * .9)

    trainset = df[:train_size]
    testset = df[train_size:]
    cl = one_run(trainset[features], get_flattened_labels(trainset), testset[features], get_flattened_labels(testset), steps)
