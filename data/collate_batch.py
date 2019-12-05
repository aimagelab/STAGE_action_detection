import torch
from utils import boxlist_ops


class BatchCollator(object):

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        actors_features = torch.cat(transposed_batch[0], dim=0)
        actors_labels = torch.cat(transposed_batch[1], dim=0)
        actors_boxes = torch.cat(transposed_batch[2], dim=0)
        actors_filenames = sum(transposed_batch[3], [])

        objects_features = torch.cat(transposed_batch[4], dim=0)
        objects_boxes = torch.cat(transposed_batch[5], dim=0)
        objects_filenames = sum(transposed_batch[6], [])

        num_actor_proposals = actors_boxes.shape[0]
        num_object_proposals = objects_boxes.shape[0]

        adj = torch.zeros((num_actor_proposals + num_object_proposals, num_actor_proposals + num_object_proposals))

        cur_actors = 0
        cur_objects = 0

        tau=1
        # populate the adj matrix in the actor-actor and actor-object sections
        for i in range(len(transposed_batch[0])):
            adj[cur_actors:cur_actors + transposed_batch[0][i].shape[0],cur_actors:cur_actors + transposed_batch[0][i].shape[0]] = boxlist_ops.boxlist_distance(transposed_batch[2][i], transposed_batch[2][i], tau=tau)
            adj[cur_actors:cur_actors + transposed_batch[0][i].shape[0], num_actor_proposals + cur_objects:num_actor_proposals + cur_objects + transposed_batch[4][i].shape[0]] = boxlist_ops.boxlist_distance(transposed_batch[2][i], transposed_batch[5][i], tau=tau)
            if i==0:
                adj[cur_actors:cur_actors + transposed_batch[0][i].shape[0], cur_actors + transposed_batch[0][i].shape[0]:cur_actors + transposed_batch[0][i].shape[0] + transposed_batch[0][i+1].shape[0]] = boxlist_ops.boxlist_distance(transposed_batch[2][i], transposed_batch[2][i+1], tau=tau)
                adj[cur_actors:cur_actors + transposed_batch[0][i].shape[0], num_actor_proposals + cur_objects + transposed_batch[4][i].shape[0]:num_actor_proposals + cur_objects + transposed_batch[4][i].shape[0] + transposed_batch[4][i+1].shape[0]] = boxlist_ops.boxlist_distance(transposed_batch[2][i], transposed_batch[5][i+1], tau=tau)
            elif i == len(transposed_batch[3]) - 1:
                adj[cur_actors:cur_actors + transposed_batch[0][i].shape[0], cur_actors - transposed_batch[0][i-1].shape[0]:cur_actors] = boxlist_ops.boxlist_distance(transposed_batch[2][i], transposed_batch[2][i-1], tau=tau)
                adj[cur_actors:cur_actors + transposed_batch[0][i].shape[0], num_actor_proposals + cur_objects - transposed_batch[4][i-1].shape[0]:num_actor_proposals + cur_objects] = boxlist_ops.boxlist_distance(transposed_batch[2][i], transposed_batch[5][i-1], tau=tau)
            else:
                adj[cur_actors:cur_actors + transposed_batch[0][i].shape[0], cur_actors + transposed_batch[0][i].shape[0]:cur_actors + transposed_batch[0][i].shape[0] + transposed_batch[0][i + 1].shape[0]] = boxlist_ops.boxlist_distance(transposed_batch[2][i], transposed_batch[2][i + 1], tau=tau)
                adj[cur_actors:cur_actors + transposed_batch[0][i].shape[0], cur_actors - transposed_batch[0][i - 1].shape[0]:cur_actors] = boxlist_ops.boxlist_distance(transposed_batch[2][i], transposed_batch[2][i - 1], tau=tau)
                adj[cur_actors:cur_actors + transposed_batch[0][i].shape[0], num_actor_proposals + cur_objects + transposed_batch[4][i].shape[0]:num_actor_proposals + cur_objects + transposed_batch[4][i].shape[0] + transposed_batch[4][i + 1].shape[0]] = boxlist_ops.boxlist_distance(transposed_batch[2][i], transposed_batch[5][i + 1], tau=tau)
                adj[cur_actors:cur_actors + transposed_batch[0][i].shape[0], num_actor_proposals + cur_objects - transposed_batch[4][i - 1].shape[0]:num_actor_proposals + cur_objects] = boxlist_ops.boxlist_distance(transposed_batch[2][i], transposed_batch[5][i - 1], tau=tau)

            cur_actors += transposed_batch[0][i].shape[0]
            cur_objects += transposed_batch[4][i].shape[0]

        # populate the adj matrix in the object-actor section
        adj[num_actor_proposals:, :num_actor_proposals] = torch.t(adj[:num_actor_proposals, num_actor_proposals:])
        cur_objects = 0

        # populate the adj matrix in the object-object section
        for i in range(len(transposed_batch[4])):
            adj[num_actor_proposals + cur_objects:num_actor_proposals + cur_objects + transposed_batch[4][i].shape[0], num_actor_proposals + cur_objects:num_actor_proposals + cur_objects + transposed_batch[4][i].shape[0]] = boxlist_ops.boxlist_distance(transposed_batch[5][i], transposed_batch[5][i], tau=tau)
            if i==0:
                adj[num_actor_proposals + cur_objects:num_actor_proposals + cur_objects + transposed_batch[4][i].shape[0], num_actor_proposals + cur_objects + transposed_batch[4][i].shape[0]:num_actor_proposals + cur_objects + transposed_batch[4][i].shape[0] + transposed_batch[4][i+1].shape[0]] = boxlist_ops.boxlist_distance(transposed_batch[5][i], transposed_batch[5][i+1], tau=tau)
            elif i == len(transposed_batch[3]) - 1:
                adj[num_actor_proposals + cur_objects:num_actor_proposals + cur_objects + transposed_batch[4][i].shape[0], num_actor_proposals + cur_objects - transposed_batch[4][i-1].shape[0]:num_actor_proposals+cur_objects] = boxlist_ops.boxlist_distance(transposed_batch[5][i], transposed_batch[5][i-1], tau=tau)
            else:
                adj[num_actor_proposals + cur_objects:num_actor_proposals + cur_objects + transposed_batch[4][i].shape[0], num_actor_proposals + cur_objects + transposed_batch[4][i].shape[0]:num_actor_proposals + cur_objects + transposed_batch[4][i].shape[0] + transposed_batch[4][i + 1].shape[0]] = boxlist_ops.boxlist_distance(transposed_batch[5][i], transposed_batch[5][i + 1], tau=tau)
                adj[num_actor_proposals + cur_objects:num_actor_proposals + cur_objects + transposed_batch[4][i].shape[0], num_actor_proposals + cur_objects - transposed_batch[4][i - 1].shape[0]:num_actor_proposals + cur_objects] = boxlist_ops.boxlist_distance(transposed_batch[5][i], transposed_batch[5][i - 1], tau=tau)

            cur_objects += transposed_batch[4][i].shape[0]

        return actors_features, actors_labels, actors_boxes, actors_filenames, objects_features, objects_boxes, objects_filenames, adj