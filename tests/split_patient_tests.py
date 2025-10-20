from split_patient import split_patients
import pandas as pd

def test_split_patient_seperate():
    tr, te = split_patients(base_path="/Code/DL/bbosis/Hongyuan-Babesiosis", labels_file="tests/Test_Labels.csv")

    tr_pids = tr["PtID"]
    te_pids = te["PtID"]

    # assert(set(tr_pids).isdisjoint(set(te_pids)), True)
    print(set(tr_pids).isdisjoint(set(te_pids)))
    return set(tr_pids).isdisjoint(set(te_pids))

test_split_patient_seperate()