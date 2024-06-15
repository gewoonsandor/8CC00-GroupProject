from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score)

x_data = ['MolWt',
          'TPSA',
          'MolLogP',
          'NumHDonors',
          'NumHAcceptors',
          'fr_NH1',
          'fr_benzene',
          'fr_phenol',
          'CalcNumRotatableBonds']

x_data_fingerprint = x_data + [f'Fingerprint_{x}' for x in range(1024)]

y_data = ['PKM2_inhibition', 'ERK2_inhibition']


def get_mol_descriptors(mol):
    mol_descriptors = {}
    for descriptor in x_data:
        try:
            if hasattr(Descriptors, descriptor):
                descriptor_func = getattr(Descriptors, descriptor)
                mol_descriptors[descriptor] = descriptor_func(mol)
            elif hasattr(rdMolDescriptors, descriptor):
                descriptor_func = getattr(rdMolDescriptors, descriptor)
                mol_descriptors[descriptor] = descriptor_func(mol)
            else:
                print(f"{descriptor} '{mol}' is not available in RDKit.")
                mol_descriptors[descriptor] = None
        except AttributeError:
            print(f"{descriptor} '{mol}' is not callable in RDKit.")
            mol_descriptors[descriptor] = None
    return mol_descriptors


def calculate_scores(predictions, test_set, name):
    # Calculate metrics for each property
    accuracy = accuracy_score(test_set, predictions)
    precision = precision_score(test_set, predictions,
                                zero_division=1)
    recall = recall_score(test_set, predictions,
                          zero_division=1)
    f1 = f1_score(test_set, predictions,
                  zero_division=1)

    print(f'{name} - Accuracy: {accuracy:.4f}, precision: {precision:.4f}, '
          f'recall: {recall:.4f}, f1: {f1:.4f}')

    return accuracy, precision, recall, f1
