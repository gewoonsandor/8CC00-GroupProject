from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score)

x_data_pkm2 = [
    "NumAromaticRings",
    "RingCount",
    "fr_thiazole",
    "BertzCT",
    "BalabanJ",
    "Chi4v",
    "AvgIpc",
    "Chi3v",
    "PEOE_VSA3",
    "Chi2v",
    "NumHAcceptors",
    "BCUT2D_MWHI",
    "VSA_EState1",
    "NumHeteroatoms",
    "VSA_EState9",
    "EState_VSA6",
    "BCUT2D_MRHI",
    "fr_sulfonamd",
    "fr_thiophene",
    "PEOE_VSA5",
    "SlogP_VSA3",
    "Chi1v",
    "SMR_VSA10",
    "BCUT2D_MWLOW",
    "SlogP_VSA5",
    "NumHDonors",
    "NHOHCount",
    "HeavyAtomMolWt",
    "NumAromaticHeterocycles",
    "SlogP_VSA6",
    "VSA_EState5",
    "EState_VSA3",
    "fr_C_O",
    "fr_C_O_noCOO",
    "fr_furan",
    "MolWt",
    "ExactMolWt",
    "FractionCSP3",
    "BCUT2D_MRLOW",
    "VSA_EState10",
    "Ipc",
    "PEOE_VSA1",
    "NOCount",
    "PEOE_VSA14",
    "MinEStateIndex",
    "SlogP_VSA12",
    "VSA_EState2",
    "SlogP_VSA1",
    "Chi0v",
    "SlogP_VSA8",
    "LabuteASA",
    "PEOE_VSA4"
]

x_data_erk2 = [
    "FpDensityMorgan1",
    "SMR_VSA9",
    "SlogP_VSA6",
    "SlogP_VSA8",
    "NumAromaticRings",
    "RingCount",
    "MolLogP",
    "fr_C_O",
    "fr_C_O_noCOO",
    "fr_nitro",
    "fr_nitro_arom",
    "PEOE_VSA4",
    "VSA_EState6",
    "SlogP_VSA5",
    "NumAromaticCarbocycles",
    "fr_benzene",
    "fr_thiazole",
    "fr_amide",
    "VSA_EState5",
    "SlogP_VSA10",
    "EState_VSA7",
    "SlogP_VSA3",
    "BertzCT",
    "BCUT2D_LOGPHI",
    "SMR_VSA7",
    "BalabanJ",
    "FpDensityMorgan2",
    "SMR_VSA5",
    "EState_VSA2",
    "Ipc",
    "MolMR",
    "HeavyAtomMolWt",
    "fr_aniline",
    "fr_aryl_methyl",
    "fr_alkyl_halide",
    "FractionCSP3",
    "MolWt",
    "ExactMolWt",
    "Chi1",
    "fr_Ar_NH",
    "fr_Nhpyrrole",
    "LabuteASA",
    "EState_VSA4"
]

x_data = list(set(x_data_erk2 + x_data_pkm2))
fingerprints = [f'Fingerprint_{x}' for x in range(1024)]

x_data_fingerprint = x_data + fingerprints
x_data_pkm2_fingerprint = x_data_pkm2 + fingerprints
x_data_erk2_fingerprint = x_data_erk2 + fingerprints


y_data = ['PKM2_inhibition', 'ERK2_inhibition']


def get_descriptors(inhibitor):
    match inhibitor:
        case 'PKM2_inhibition':
            return x_data_pkm2_fingerprint
        case 'ERK2_inhibition':
            return x_data_erk2_fingerprint


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
