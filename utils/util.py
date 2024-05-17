import torch
import os

def save_model(name,model,save_path):
    torch.save(dict(params=model.state_dict()), os.path.join(save_path, name + '.pth'))

#存储结果
def create_file(file_path, file_name):
    num = 1
    file_list = os.listdir(file_path)
    for i in file_list:
        if os.path.exists(file_path + './%s' % (file_name + '_' + str(num))):
            num += 1
    os.makedirs(file_path + './%s' % (file_name + '_' + str(num)))
    path = os.path.join(file_path, (file_name + '_%d' % num))
    exp = file_name + '_' + str(num)
    return path,exp