from __future__ import print_function, division
import os
import json
import time

from utils import command_parser
from utils.class_finder import model_class, agent_class
from main_eval import main_eval
from tqdm import tqdm
from tabulate import tabulate

from tensorboardX import SummaryWriter

os.environ["OMP_NUM_THREADS"] = "1"


def single_eval(args=None):
    if args is None:
        args = command_parser.parse_arguments()

    create_shared_model = model_class(args.model)
    init_agent = agent_class(args.agent_type)

    args.phase = 'eval'
    args.episode_type = 'TestValEpisode'
    args.test_or_val = 'test'

    # if args.num_category != 60:
    #     args.detection_feature_file_name = 'det_feature_{}_categories.hdf5'.format(args.num_category)

    start_time = time.time()
    local_start_time_str = time.strftime(
        '%Y_%m_%d_%H_%M_%S', time.localtime(start_time)
    )

    tb_log_dir = args.log_dir + "/" + args.title + '_' + args.phase + '_' + local_start_time_str
    log_writer = SummaryWriter(log_dir=tb_log_dir)

    checkpoint = args.load_model

    model = os.path.join(args.save_model_dir, checkpoint)
    args.load_model = model

    # run eval on model
    # args.test_or_val = "val"
    main_eval(args, create_shared_model, init_agent)


if __name__ == "__main__":
    single_eval()