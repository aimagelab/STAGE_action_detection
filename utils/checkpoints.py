import torch
import os


def tag_last_checkpoint(output_dir, last_filename):
    save_file = os.path.join(output_dir, "last_checkpoint")
    with open(save_file, "w") as f:
        f.write(last_filename)


def get_checkpoint_file(output_dir):
       save_file = os.path.join(output_dir, "last_checkpoint")
       try:
           with open(save_file, "r") as f:
               last_saved = f.read()
               last_saved = last_saved.strip()
       except IOError:
           last_saved = ""
       return last_saved


def save(name, model, optimizer, output_dir, epoch, iteration):

    optimizer_state = optimizer.state_dict()

    data = {"optimizer_state": optimizer_state,
            "epoch": epoch, "iteration": iteration}
    for model_name, model_obj in model.items():
        data[model_name] = model_obj.state_dict()

    save_file = os.path.join(output_dir, "{}.pth".format(name))

    print("Saving checkpoint to {}".format(save_file))
    torch.save(data, save_file)
    tag_last_checkpoint(output_dir, save_file)


def load(output_dir):
    if os.path.exists(os.path.join(output_dir, "last_checkpoint")):
        f = get_checkpoint_file(output_dir)
    else:
        return {}
    print("Loading checkpoint from {}".format(f))
    checkpoint = torch.load(f, map_location=torch.device("cpu"))
    data = {}
    data["optimizer_state"] = checkpoint.pop("optimizer_state")
    data["epoch"] = checkpoint.pop("epoch")
    data["iteration"] = checkpoint.pop("iteration")
    data["model_state"] = checkpoint.pop("model_state")
    return data