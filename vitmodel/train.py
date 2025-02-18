import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


from my_dataset import MyDataSet
from vit_model import vit_base_patch16_224_in21k as create_model
from utils import read_split_data, train_one_epoch, evaluate
import wandb


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    wandb.init(project="Vision_Transformer", config=args)

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5], [0.5])]),
        "val": transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5], [0.5])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])
    
    # 打印训练数据集和验证数据集的第一个图像的维度
    train_image, train_label = train_dataset[0]
    val_image, val_label = val_dataset[0]
    print(f"Training image shape: {train_image.shape}")
    print(f"Validation image shape: {val_image.shape}")

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)

    if args.if_use_custom_weights: # 使用自定义预训练权重
        assert os.path.exists(args.custom_weights), "custom weights file: '{}' not exist.".format(args.custom_weights)
        model.load_state_dict(torch.load(args.custom_weights, map_location=device))
        print(f"Loaded custom pretrained weights from {args.custom_weights}")
    
    else: # 使用官方预训练权重
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)

        # 调整第一层的权重
        if 'patch_embed.proj.weight' in weights_dict:
            pretrained_weights = weights_dict['patch_embed.proj.weight']
            new_weights = pretrained_weights.mean(dim=1, keepdim=True)
            weights_dict['patch_embed.proj.weight'] = new_weights

        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))


    # if args.freeze_layers: # 冻结权重
    #     for name, para in model.named_parameters():
    #         # 除head, pre_logits外，其他权重全部冻结
    #         if "head" not in name and "pre_logits" not in name:
    #             para.requires_grad_(False)
    #         else:
    #             print("training {}".format(name))


    # 确保所有层的参数都没有被冻结
    for name, param in model.named_parameters():
        param.requires_grad = True

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_f1 = 0.0
    epochs_no_improve = 0
    early_stop_patience = 8

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc, train_f1, train_recall, train_precision, train_cm = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_acc, val_f1, val_recall, val_precision, val_cm = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "train_f1", "train_recall", "train_precision", 
        "val_loss", "val_acc", "val_f1", "val_recall", "val_precision", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], train_f1, epoch)
        tb_writer.add_scalar(tags[3], train_recall, epoch)
        tb_writer.add_scalar(tags[4], train_precision, epoch)
        tb_writer.add_scalar(tags[5], val_loss, epoch)
        tb_writer.add_scalar(tags[6], val_acc, epoch)
        tb_writer.add_scalar(tags[7], val_f1, epoch)
        tb_writer.add_scalar(tags[8], val_recall, epoch)
        tb_writer.add_scalar(tags[9], val_precision, epoch)
        tb_writer.add_scalar(tags[10], optimizer.param_groups[0]["lr"], epoch)

        # Log metrics to W&B
        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "train_recall": train_recall,
            "train_precision": train_precision,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_recall": val_recall,
            "val_precision": val_precision,
            "learning_rate": optimizer.param_groups[0]["lr"]
        })

        # Check for early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), "./weights/best_model.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="../data/small_xray_crop")
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='./vit_base_patch16_224_in21k.pth',
                        help='initial weights path')
# 是否使用自定义预训练权重
    parser.add_argument('--if-use-custom-weights', type=bool, default=True, help='use custom pretrained weights')
    # 自定义预训练权重路径
    parser.add_argument('--custom-weights', type=str, default='./model-19.pth', help='path to custom pretrained weights')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)