import os, argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import data_loader.augmentation as module_tsf
import model.model as module_arch
from parse_config import ConfigParser
import torch.nn.functional as F
import pandas as pd
from utils import prepare_device

def main(config):
    logger = config.get_logger('predict')

    # setup data_loader instances
    data_loader = data_loader = config.init_obj('data_loader', module_data)
    
    # build model architecture
    model = config.init_obj('arch', module_arch)
    transforms = config.init_obj('transformer', module_tsf)
    logger.info('Loading checkpoint: {} ...'.format(config.resume))

    checkpoint = torch.load(config.resume) # load pth model
    state_dict = checkpoint['state_dict']
    device, device_ids = prepare_device(config['n_gpu'])
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()
    
    preds = torch.tensor([]).to(device)
    probs = torch.tensor([]).to(device)
    names = []
    with torch.no_grad():
        for data, _ in tqdm(data_loader):
            mask = data[1]
            name = data[0][0]
            names += name
            if transforms:
                data = transforms(data[0])
            else:
                data = data[0]

            data, mask = data.to(device), mask.to(device)
            output = model(data, mask)
            prob = F.softmax(output, dim=1)
            prob = prob.to(device)
            pred = torch.argmax(prob, dim=1).reshape(len(name),-1)
            pred = pred.to(device)
            probs = torch.cat((probs, prob), 0)
            preds = torch.cat((preds, pred), 0)
            del prob
            del pred
            del output
            torch.cuda.empty_cache()
    
    result = torch.cat((probs, preds), 1)
    
    # write to tsv file
    dt = pd.DataFrame(result.cpu().numpy(), columns=['Not_TSP_probability', 'TSP_probability', 'Predicted_label'])
    dt['ID'] = names
    dt = dt[['ID', 'Not_TSP_probability', 'TSP_probability', 'Predicted_label']]
    dt.to_csv(config['output'], sep="\t", index=False)

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
