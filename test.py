from data.ava_dataset import AVADataset
import torch
from utils import checkpoints
import os
import argparse
from data.collate_batch import BatchCollator
from models.model import Stage
import csv
from tqdm import tqdm
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--actors_dir", type=str, help="path to the directory containing actors features")
parser.add_argument("--objects_dir", type=str, help="path to the file containing objects features")
parser.add_argument("--output_dir", type=str, help="path to the directory where checkpoints will be stored")
parser.add_argument("--batch_size", type=int)
parser.add_argument("--n_workers", type=int)
parser.add_argument("--num_classes", type=int, default=81)
parser.add_argument("--actors_features_size", type=int, default=1024)
parser.add_argument("--objects_features_size", type=int, default=2048)
parser.add_argument("--n_heads", type=int, default=4, help="only 2 or 4 heads supported at the moment")


def main():
    args = parser.parse_args()

    num_classes = args.num_classes
    output_dir = args.output_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    assert os.path.isdir(output_dir)

    ava_val = AVADataset(split='val', videodir=args.actors_dir, objectsfile=args.objects_dir)
    data_loader_val = torch.utils.data.DataLoader(ava_val, batch_size=args.batch_size, num_workers=args.n_workers, collate_fn=BatchCollator(), shuffle=False)

    model = Stage(num_classes, args.actors_features_size, args.objects_features_size, args.n_heads)

    model.eval()
    model.to(device)

    checkpoint = checkpoints.load(output_dir)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=False)
    else:
        sys.exit('No checkpoint found!')

    with torch.no_grad():
        with open(os.path.join(output_dir, "results.csv"), mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for iteration_val, (actors_features, actors_labels, actors_boxes, actors_filenames, objects_features, objects_boxes, objects_filenames, adj) in enumerate(tqdm(data_loader_val), 1):
                actors_features = actors_features.to(device)
                actors_labels = actors_labels.to(device)
                actors_boxes = actors_boxes.to(device)
                objects_features = objects_features.to(device)
                objects_boxes = objects_boxes.to(device)
                adj = adj.to(device)

                pred, loss = model(actors_features, actors_labels, actors_boxes, objects_features, objects_boxes, adj)

                for i, prop in enumerate(pred):
                    classes = torch.nonzero(prop)
                    for c in classes:
                        if int(c) != 0:  # do not consider background class
                            csv_writer.writerow(
                                [actors_filenames[i][0], str(int(actors_filenames[i][1])),
                                 str(actors_boxes[i, 0].item()), str(actors_boxes[i, 1].item()),
                                 str(actors_boxes[i, 2].item()), str(actors_boxes[i, 3].item()),
                                 int(c),
                                 prop[int(c)].item()])
                csv_file.flush()

            csv_file.close()


if __name__ == '__main__':
    main()

