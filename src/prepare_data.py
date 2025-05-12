import numpy as np
import pandas as pd
from reaction_data import get_graph_data
from utils import configure_warnings_and_logs
from sklearn.model_selection import train_test_split

configure_warnings_and_logs(ignore_warnings=True)


def prepare_data(args):
    inference_reaction=np.load(args.Data_folder + args.data_inference)
    data = pd.read_csv(args.Data_folder + args.data_csv)

    rsmi_list = data[args.reaction_column].values
    rmol_max_cnt = np.max([smi.split(">>")[0].count(".") + 1 for smi in inference_reaction])
    pmol_max_cnt = np.max([smi.split(">>")[1].count(".") + 1 for smi in inference_reaction])

    dim_y=data[args.y_column].nunique()
    y_pseudo=np.eye(dim_y,dtype="uint8")[[0 for _ in range(len(inference_reaction))]]
    filename=args.Data_folder + args.npz_inference + "/inference.npz" 

    get_graph_data(
        args,
        inference_reaction,
        y_pseudo,
        filename,
        rmol_max_cnt,
        pmol_max_cnt,
    )
