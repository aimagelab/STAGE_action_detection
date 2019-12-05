import torch
from models.gat import GAT
from torch import nn


class Stage(torch.nn.Module):

    def __init__(self, num_classes, actors_features_size, objects_features_size, n_heads):
        super(Stage, self).__init__()

        self.num_classes = num_classes
        self.actors_features_size = actors_features_size
        self.objects_features_size = objects_features_size
        self.n_heads = n_heads

        if self.objects_features_size > self.actors_features_size:
            self.obj_reducer = nn.Linear(in_features=self.objects_features_size, out_features=self.actors_features_size)

        self.gat1 = GAT(self.actors_features_size + 4, int(self.actors_features_size/self.n_heads) + int(4/self.n_heads), 0.5, 0.2, self.n_heads) #we add 4 because we are going to add h,w,xc,yc to the channel axis
        self.gat_fc = nn.Linear(in_features=self.actors_features_size +4, out_features=self.actors_features_size +4)
        self.l_norm = nn.LayerNorm(self.actors_features_size +4)

        self.gat2 = GAT(self.actors_features_size + 4, int(self.actors_features_size/self.n_heads) + int(4/self.n_heads), 0.5, 0.2, self.n_heads)
        self.gat_fc2 = nn.Linear(in_features=self.actors_features_size +4, out_features=self.actors_features_size +4)
        self.l_norm2 = nn.LayerNorm(self.actors_features_size +4)

        self.logits = nn.Linear(in_features=self.actors_features_size +4, out_features=self.num_classes)


    def forward(self, actors_features, actors_labels, actors_boxes, objects_features, objects_boxes, adj):
        #compute h, w, xc, yc for each actor/object
        actors_h = actors_boxes[:, 3] - actors_boxes[:, 1]
        objects_h = objects_boxes[:, 3] - objects_boxes[:, 1]
        actors_w = actors_boxes[:, 2] - actors_boxes[:, 0]
        objects_w = objects_boxes[:, 2] - objects_boxes[:, 0]
        actors_centers_x = ((actors_boxes[:, 2] - actors_boxes[:, 0]) / 2) + actors_boxes[:, 0]
        actors_centers_y = ((actors_boxes[:, 3] - actors_boxes[:, 1]) / 2) + actors_boxes[:, 1]
        objects_centers_x = ((objects_boxes[:, 2] - objects_boxes[:, 0]) / 2) + objects_boxes[:, 0]
        objects_centers_y = ((objects_boxes[:, 3] - objects_boxes[:, 1]) / 2) + objects_boxes[:, 1]

        with torch.no_grad():
            actors_features = torch.mean(torch.mean(torch.mean(actors_features, dim=2), dim=-1), dim=-1)
        actors_features = torch.cat((actors_features, actors_h.unsqueeze(1), actors_w.unsqueeze(1), actors_centers_x.unsqueeze(1), actors_centers_y.unsqueeze(1)), dim=1)

        if self.objects_features_size > self.actors_features_size:
            objects_features = nn.functional.relu(self.obj_reducer(objects_features))

        objects_features = torch.cat((objects_features, objects_h.unsqueeze(1), objects_w.unsqueeze(1), objects_centers_x.unsqueeze(1), objects_centers_y.unsqueeze(1)), dim=1)

        all_features = torch.cat((actors_features, objects_features), dim=0)

        gat_pred = self.gat1(all_features, adj)
        all_features = all_features + self.gat_fc(gat_pred)
        all_features = self.l_norm(all_features)

        gat_pred = self.gat2(all_features, adj)
        all_features = all_features + self.gat_fc2(gat_pred)
        all_features = self.l_norm2(all_features)

        pred = self.logits(all_features[:actors_features.shape[0], :])

        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, actors_labels)

        pred = torch.sigmoid(pred)
        return pred, loss
