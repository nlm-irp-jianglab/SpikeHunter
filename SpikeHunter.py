import os, argparse
import warnings
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import data_loader.augmentation as module_tsf
import model.model as module_arch
from parse_config import ConfigParser
import torch.nn.functional as F
import pandas as pd
from utils import prepare_device

# Suppress FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning)

def main(config):
    # Speed up convolutions/GEMMs on fixed input shapes
    torch.backends.cudnn.benchmark = True
    # Prefer TF32 on Ampere/Hopper GPUs for faster matmuls
    torch.set_float32_matmul_precision('medium')

    logger = config.get_logger('predict')

    # skip IDs that are already predicted so we can resume incomplete runs
    output_path = config['output']
    existing_ids = set()
    append_mode = False
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path, sep="\t")
            existing_ids = set(existing_df['ID'].astype(str))
            append_mode = True
            logger.info(f"Found existing predictions: {len(existing_ids)} IDs. Skipping already processed entries.")
        except Exception as e:
            logger.warning(f"Could not read existing output file {output_path}: {e}. Recomputing all entries.")

    # setup data_loader instances, filtering skips during construction
    dl_cfg = config['data_loader']
    dl_args = dl_cfg['args']
    DataLoaderCls = getattr(module_data, dl_cfg['type'])
    data_loader = DataLoaderCls(skip_ids=existing_ids, **dl_args) if existing_ids else config.init_obj('data_loader', module_data)

    skipped = getattr(data_loader, 'skipped', 0)
    if skipped:
        logger.info(f"Skipped {skipped} entries already predicted; {len(data_loader.dataset)} remaining.")
    else:
        logger.info("No existing prediction file found or no IDs skipped. Predicting full dataset.")

    if len(data_loader.dataset) == 0:
        logger.info("All sequences already predicted. Exiting.")
        return
    
    # build model architecture
    model = config.init_obj('arch', module_arch)
    transforms = config.init_obj('transformer', module_tsf)
    logger.info('Loading checkpoint: {} ...'.format(config.resume))

    checkpoint = torch.load(config.resume, weights_only=False) # load pth model
    state_dict = checkpoint['state_dict']
    for key in list(state_dict.keys()):
        if 'module.' in key:
            state_dict[key.replace('module.', '')] = state_dict[key]
            del state_dict[key]

    device, device_ids = prepare_device(config['n_gpu'])
    model.load_state_dict(state_dict)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.eval()
    
    write_header = not append_mode
    processed = 0
    use_autocast = torch.cuda.is_available()

    with torch.inference_mode():
        for data, _ in tqdm(data_loader):
            mask = data[1]
            name = data[0][0]
            if transforms:
                data = transforms(data[0])
            else:
                data = data[0]

            data = data.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=use_autocast):
                output = model(data, mask)
                prob = F.softmax(output, dim=1)
            prob = prob.to(device)
            pred = torch.argmax(prob, dim=1).reshape(len(name),-1)
            pred = pred.to(device)
            # append batch results so interrupted runs can resume later
            batch_df = pd.DataFrame({
                'ID': list(name),
                'Not_TSP_probability': prob[:, 0].detach().float().cpu().numpy(),
                'TSP_probability': prob[:, 1].detach().float().cpu().numpy(),
                'Predicted_label': pred.detach().cpu().numpy().reshape(len(name))
            })
            batch_df.to_csv(output_path, sep="\t", index=False, mode='a' if append_mode else 'w', header=write_header)
            append_mode = True
            write_header = False
            processed += len(name)
            del prob
            del pred
            del output
    logger.info(f"Finished writing {processed} new predictions to {output_path}.")

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='SpikeHunter')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    config = ConfigParser.from_args(args)
    main(config)
