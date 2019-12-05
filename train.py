from data.ava_dataset import AVADataset
import torch
from utils import checkpoints
from tensorboardX import SummaryWriter
import os
import errno
import argparse
from data.collate_batch import BatchCollator
from models.model import Stage


parser = argparse.ArgumentParser()
parser.add_argument("--actors_dir", type=str, help="path to the directory containing actors features")
parser.add_argument("--objects_file", type=str, help="path to the file containing objects features")
parser.add_argument("--output_dir", type=str, help="path to the directory where checkpoints will be stored")
parser.add_argument("--log_tensorboard_dir", type=str, help="path to the directory where tensorboard logs will be stored")
parser.add_argument("--batch_size", type=int)
parser.add_argument("--n_workers", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--num_classes", type=int, default=81)
parser.add_argument("--actors_features_size", type=int, default=1024)
parser.add_argument("--objects_features_size", type=int, default=2048)
parser.add_argument("--n_heads", type=int, default=4, help="only 2 or 4 heads supported at the moment")
parser.add_argument("--impose_lr", type=float, default=0.0, help="if not zero, will impose the specified learning rate to all the optimizer's parameters")
parser.add_argument("--n_epochs", type=int, default=50)


def main():
    args = parser.parse_args()

    num_classes = args.num_classes
    start_epoch = 1
    start_iter = 1
    output_dir = args.output_dir
    tensorboard_dir = args.log_tensorboard_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        os.makedirs(output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    ava_train = AVADataset(split='train', videodir=args.actors_dir, objectsfile=args.objects_file)
    data_loader_train = torch.utils.data.DataLoader(ava_train, batch_size=args.batch_size, num_workers=args.n_workers, collate_fn=BatchCollator(), shuffle=False)

    writer = SummaryWriter(tensorboard_dir)

    model = Stage(num_classes, args.actors_features_size, args.objects_features_size, args.n_heads)

    model.train()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    checkpoint = checkpoints.load(output_dir)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if args.impose_lr != 0.0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.impose_lr
        start_epoch = checkpoint["epoch"]
        start_iter = checkpoint["iteration"] + 1
    else:
        print("No checkpoint found: initializing model from scratch")

    for current_epoch in range(start_epoch, args.n_epochs + 1):

        data_loader_train.dataset.shuffle_filename_blocks(args.batch_size, current_epoch)

        for iteration, (actors_features, actors_labels, actors_boxes, actors_filenames, objects_features, objects_boxes, objects_filenames, adj) in enumerate(data_loader_train, start_iter):
            actors_features = actors_features.to(device)
            actors_labels = actors_labels.to(device)
            actors_boxes = actors_boxes.to(device)
            objects_features = objects_features.to(device)
            objects_boxes = objects_boxes.to(device)
            adj = adj.to(device)

            pred, loss = model(actors_features, actors_labels, actors_boxes, objects_features, objects_boxes, adj)

            loss = torch.mean(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % 20 == 0:
                writer.add_scalar('train/class_loss', loss, (current_epoch-1) * len(data_loader_train) + iteration)
                print("epoch: " + str(current_epoch) + ", iter: " + str(iteration) + "/" + str(len(data_loader_train)) + ", lr: " + str(optimizer.param_groups[0]["lr"]) + ", class_loss: " + str(loss.item()))

            if iteration >= len(data_loader_train):
                print("Epoch " + str(current_epoch) + "ended.")
                start_iter = 1
                models = {"model_state": model}
                checkpoints.save("model_ep_{:03}_iter_{:07d}".format(current_epoch, iteration), models, optimizer, output_dir, current_epoch, iteration)

                break

    checkpoints.save("final_model", models, optimizer, output_dir, current_epoch, iteration)


if __name__ == '__main__':
    main()
