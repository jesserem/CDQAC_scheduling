import torch
import os
import pandas as pd

def read_basic(f_path: str, sep: str = ' ', device: str = 'cpu'):
    """
    Load the basic information about a JSP instance.

    Structure of the file:
        1. num_jobs num_machines
        2. instance matrix (each row is a job)
        3. makespan
    """
    with open(f_path) as f:
        # Load the shape
        shape = next(f).split(sep)
        n = int(shape[0])
        m = int(shape[1])
        name = f_path.rsplit('/', 1)[1].rsplit('.', 1)[0]

        # Load the instance
        instance = torch.empty((n, 2 * m), dtype=torch.float32, device=device)
        for j in range(n):
            instance[j] = torch.tensor(
                [float(x) for x in next(f).split(sep) if x and not x.isspace()],
                device=device
            )

        # Load the makespan of a reference solution, if any
        ms = 1.
        try:
            _ms = next(f)
            if _ms != '':
                ms = float(_ms)
        except StopIteration:
            pass
    return name, n, m, instance, ms

def analyze_folder(data_dict, folder_path):
    """
    Analyze the folder containing JSP instances.
    """
    files = os.listdir(folder_path)

    for file in files:
        if file.endswith('.jsp'):
            file_path = os.path.join(folder_path, file)
            name, n, m, instance, ms = read_basic(file_path)

            benchname, dataname = name.split('\\')
            if benchname == "LA":
                benchname = "Lawrence"
            elif benchname == "TA":
                benchname = "Taillard"
            elif benchname == "DMU":
                benchname = "Demirkol"
            else:
                raise RuntimeError(f"Unknown benchmark: {benchname}")
            file_n = file.split('.')[0]
            data_dict["benchname"].append(benchname)
            data_dict["dataname"].append(dataname)
            data_dict["filename"].append(file_n)
            data_dict["n_j"].append(n)
            data_dict["n_m"].append(m)
            data_dict["ub"].append(int(ms))

def main():
    d_dict = {
        "benchname": [],
        "dataname": [],
        "filename": [],
        "n_j": [],
        "n_m": [],
        "ub": [],

    }
    analyze_folder(d_dict, './data/JSP/LA')
    analyze_folder(d_dict, './data/JSP/DMU')
    analyze_folder(d_dict, './data/JSP/TA')
    df_bench = pd.DataFrame(d_dict)
    df_bench.to_csv('./data/JSP/benchmark_results.csv', index=False)

main()