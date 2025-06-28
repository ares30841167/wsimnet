import os
import glob

from utils.reader import load_json_file


# Split the company labels into train and validate company labels
def split_train_and_validate_company_labels(dataset_path: str) -> tuple[dict[str, str], dict[str, str]]:
    # Load the company labels
    company_labels = load_json_file(f'{dataset_path}/company_labels.json')

    # Get all the website names that belong to the validation set
    train_website = glob.glob(os.path.join(
        f'{dataset_path}/train', '*.gexf'))
    train_website = set([os.path.splitext(os.path.basename(f))[
        0] for f in train_website])

    # Get all the website names that belong to the validation set
    validate_website = glob.glob(os.path.join(
        f'{dataset_path}/validate', '*.gexf'))
    validate_website = set([os.path.splitext(os.path.basename(f))[
                           0] for f in validate_website])

    # Initial variable
    train_company_labels = {}
    validate_company_labels = {}

    # Classify the labels
    for site_name, label in company_labels.items():
        if (site_name in train_website):
            train_company_labels[site_name] = label
        elif (site_name in validate_website):
            validate_company_labels[site_name] = label

    return train_company_labels, validate_company_labels


# Split the company labels into train, validate and test company labels
def split_company_labels(dataset_path: str) -> tuple[dict[str, str], dict[str, str]]:
    # Load the company labels
    company_labels = load_json_file(f'{dataset_path}/company_labels.json')

    # Get all the website names that belong to the validation set
    train_website = glob.glob(os.path.join(
        f'{dataset_path}/train', '*.gexf'))
    train_website = set([os.path.splitext(os.path.basename(f))[
        0] for f in train_website])

    # Get all the website names that belong to the validation set
    validate_website = glob.glob(os.path.join(
        f'{dataset_path}/validate', '*.gexf'))
    validate_website = set([os.path.splitext(os.path.basename(f))[
                           0] for f in validate_website])
    
    # Get all the website names that belong to the test set
    test_website = glob.glob(os.path.join(
        f'{dataset_path}/test', '*.gexf'))
    test_website = set([os.path.splitext(os.path.basename(f))[
                           0] for f in test_website])

    # Initial variable
    train_company_labels = {}
    validate_company_labels = {}
    test_company_labels = {}

    # Classify the labels
    for site_name, label in company_labels.items():
        if (site_name in train_website):
            train_company_labels[site_name] = label
        elif (site_name in validate_website):
            validate_company_labels[site_name] = label
        elif (site_name in test_website):
            test_company_labels[site_name] = label

    return train_company_labels, validate_company_labels, test_company_labels
