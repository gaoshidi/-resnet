import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import resnet34
from torch.utils.tensorboard import SummaryWriter

write = SummaryWriter()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        # resize将输入图片的长宽比固定，将最小边缩放到256
        "val": transforms.Compose([transforms.Resize(256),
                                   # 使用中心裁剪得到224
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # data_root = os.path.abspath(os.path.join(os.getcwd(), "D:/pythonProject/flowClassify"))  # get data root path
    data_root = os.path.abspath(os.path.join(os.getcwd(), "D:/论文/为发论文做实验/数据集/哥本哈根狗类数据集/dataset"))  # get data root path
    # image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    image_path = os.path.join(data_root, "9vs1")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 64
    epochs = 15
    best_acc = 0.0
    save_path = './resNet34.pth'
    # 把日志写到一个地方
    writer = SummaryWriter(log_dir='./log/')
    # tensorboard --logdir=训练内容的文件夹名称

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    
    net = resnet34()
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth

    # 博主还有一种载入模型的方式，先torch.load(model_weight_path, map_location='cpu')把权重放到内存中
    # 然后删掉全连接层数据
    # 这样的好处是，一开始初始化的时候就能得到类别=5，而不是imagnet中的1000种
    model_weight_path = "./resnet34-pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'),strict=False)
    # for param in net.parameters():
    #     param.requires_grad = False
    # change fc layer structure
    in_channel = net.fc.in_features
    # 这里指定多少种类别
    net.fc = nn.Linear(in_channel, 120)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    # 初始学习率————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    optimizer = optim.Adam(params, lr=0.00003)


    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            # 全连接层的输出
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            # 计算误差之后进行反向传播，计算损失梯度信息
            loss.backward()
            # 更新参数
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        X.append(epoch + 1)
        Y.append(val_accurate)
        Z.append(running_loss / train_steps)
        write.add_scalar("损失", epoch + 1, running_loss / train_steps)
        write.add_scalar("准确度", epoch + 1, val_accurate)
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    X, Y, Z = [], [], []
    main()
    plt.plot(X, Y, color="black", lineWidth=2.0, LineStyle="-")
    plt.show()
    plt.plot(X, Z, color="pink", lineWidth=2.0, LineStyle="-")
    plt.show()
