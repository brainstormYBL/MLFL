import torch

from Models.Models import LinearRegression
from Utils.Parameters import parameters
from Utils.ProcessData import ProcessData

if __name__ == "__main__":
    par = parameters()
    path_data_set = r"/Volumes/科研/4.AI/2.联邦学习/2.Project/1.FedAvg/Data/heart.csv"
    data_holder = ProcessData(path_data_set, par.ratio_train, par.num_client)
    model_t = LinearRegression(dim_input=data_holder.dim_feature, dim_output=data_holder.dim_label)
    model_t.load_state_dict(torch.load('model.pth'))
    model_t.eval()

    loss = torch.nn.MSELoss()

    y_pre = model_t(data_holder.all_feature_tensor)
    loss_val = loss(y_pre, data_holder.all_label_tensor)
    print(loss_val.item())
