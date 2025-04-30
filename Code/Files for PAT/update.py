import os
import csv
import re
import argparse
from typing import Literal, List, Dict
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from config import cfg
from model import make_model
from utils.logger import setup_logger
from utils.re_ranking import re_ranking
from data.build_DG_dataloader import build_reid_test_loader
from processor.ori_vit_processor_with_amp import do_inference as do_inf
from processor.part_attention_vit_processor import do_inference as do_inf_pat


def extract_feature(model: nn.Module, dataloaders: DataLoader, subset: Literal['query', 'gallery'], filename_pattern: str = r'\.') -> np.ndarray:
    '''
    Parameters:
        model: resnet featurizer
        dataloaders: for everything
        subset: to which suibset should the featurization should be restricted
        filename_pattern: matches filename with `re.search`. The default value includes all images
    Return:
        feature vector shape [<number of images in subset, 768]
    '''
    assert isinstance(model, nn.Module)
    assert isinstance(dataloaders, DataLoader)
    assert type(num_query) == int
    with torch.no_grad():
        features = torch.FloatTensor(0, 768).cuda()
        count = 0
        img_path = []
        for data in dataloaders:
            images, _, _, filenames, metadatas = data.values()

            # Select only the images that belong to the desired subset
            subsets: List[Literal['query', 'gallery']] = metadatas['q_or_g']
            mask = torch.tensor([s == subset for s in subsets], dtype=torch.bool)
            images = images[mask]
            filenames = list(np.array(filenames)[mask])
            assert len(images) == sum(1 for s in subsets if s == subset)

            # Select only the images that match the filename pattern
            mask = torch.tensor([bool(re.search(filename_pattern, fn)) for fn in filenames], dtype=torch.bool)
            images = images[mask]
            filenames = list(np.array(filenames)[mask])
            assert len(images) == sum(mask), "Selection count mismatch"

            n, c, h, w = images.size()
            if n==0:
                continue

            count += n
            ff = torch.FloatTensor(n, 768).zero_().cuda()  # 2048 is pool5 of resnet
            for i in range(2):
                input_img = images.cuda()
                outputs = model(input_img)
                f = outputs.float()
                ff = ff + f
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            features = torch.cat([features, ff], 0)
        assert features.shape[1] == 768
        return features.cpu().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Training")
    parser.add_argument(
        "--config_file", default="./config/PAT.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument(
        "--track", default="./config/PAT.yml", help="path to config file", type=str
    )
    args = parser.parse_args()
    num_gallery = 10#?    # I don't know how to check the expected number of galery images

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("PAT", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    model = make_model(cfg, cfg.MODEL.NAME, 0,0,0)
    model.load_param(cfg.TEST.WEIGHT)

    for testname in cfg.DATASETS.TEST:
        #print('>>>>>>>>>>>>>>>>>>>>>>>', testname)
        val_loader, num_query = build_reid_test_loader(cfg, testname)
        if cfg.MODEL.NAME == 'part_attention_vit':
            do_inf_pat(cfg, model, val_loader, num_query)
        else:
            do_inf(cfg, model, val_loader, num_query)
    
        #print(type(model))
        #print(type(val_loader))
        #print(type(num_query))

    qf = extract_feature(model, val_loader, subset='query', filename_pattern = r'^(?!.*refinement).*$')
    #qf = extract_feature(model, val_loader, subset='query')
    qf_A = extract_feature(model, val_loader, subset='query', filename_pattern = r'_refinement_A\.')
    qf_B = extract_feature(model, val_loader, subset='query', filename_pattern = r'_refinement_B\.')
    qf_C = extract_feature(model, val_loader, subset='query', filename_pattern = r'_refinement_C\.')
    gf = extract_feature(model, val_loader, subset='gallery')
    

    #assert qf.shape[0] == num_query
    #assert qf_A.shape[0] == num_query
    #assert qf_B.shape[0] == num_query
    #assert qf_B.shape[0] == num_query
    #assert gf.shape[0] >= num_query
    print('????????????????????????????????', gf.shape)

    #np.save("./qf.npy", qf)
    #np.save("./gf.npy", gf)
    q_g_dist = np.dot(qf, np.transpose(gf)) # TODO: This one will have to be calculated 3 times, and averaged out
    q_q_dist = np.dot(qf, np.transpose(qf))
    g_g_dist = np.dot(gf, np.transpose(gf))

    print('qf', qf.shape)
    print('qf_A', qf_A.shape)
    print('qf_B', qf_B.shape)
    print('qf_C', qf_C.shape)
    #assert 8==0, "UHULLL"



    print(f'Query_Gallery_dist = {q_g_dist.shape}')
    print(f'Query_Query_dist = {q_q_dist.shape}')
    print(f'Galery_Galery_dist = {g_g_dist.shape}')
    re_rank_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=10)  # TODO: original K1 was 20, but for the reduced dataset has to be 10

    indices = np.argsort(re_rank_dist, axis=1)[:, :100]

    m, n = indices.shape
    # # print('m: {}  n: {}'.format(m, n))
    with open(args.track, 'wb') as f_w:
        for i in range(m):
            write_line = indices[i] + 1
            write_line = ' '.join(map(str, write_line.tolist())) + '\n'
            f_w.write(write_line.encode())


    lista_nombres = ["{:06d}.jpg".format(i) for i in range(1, len(indices) + 1)]
    output_path = args.track.split(".txt")[0] + "_submission.csv"

    with open(output_path, 'w', newline='') as archivo_csv:
        csv_writter = csv.writer(archivo_csv)
        csv_writter.writerow(['imageName', 'Corresponding Indexes'])
        for numero, track in zip(lista_nombres, indices):
            track_str = ' '.join(map(str, track + 1))
            csv_writter.writerow([numero, track_str])
