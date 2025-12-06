import os
import torch
import numpy as np

from model import MODEL_DICT
from trainers import Trainer
from utils import EarlyStopping, check_path, set_seed, parse_args, set_logger
from dataset import get_seq_dic, get_dataloader, get_rating_matrix
from model.rl import get_cache_world_model
# from memory_profiler import profile
from pympler import tracker


# @profile
def main():

    args = parse_args()
    log_path = os.path.join(args.output_dir, args.train_name + '.log')
    logger = set_logger(log_path)

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    is_cache = args.is_cache.lower() == 'true'
    
    if not is_cache:
        seq_dic, max_item, num_users = get_seq_dic(args)
    else:
        seq_dic, max_item, num_users = get_seq_dic(args, min_len=args.num_future_steps)
    if not (hasattr(args, 'item_size') and args.item_size > 0):
        args.item_size = max_item + 1
    args.num_users = num_users + 1

    args.checkpoint_path = os.path.join(args.output_dir, args.train_name + '.pt')
    args.same_target_path = os.path.join(args.data_dir, args.data_name+'_same_target.npy')
    train_dataloader, (train_dataset, test_dataset) = get_dataloader(args, seq_dic, is_cache)

    logger.info(str(args))
    model = MODEL_DICT[args.model_type.lower()](args=args)
    logger.info(model)


    if not args.is_rl:
        args.valid_rating_matrix, args.test_rating_matrix = get_rating_matrix(args.data_name, seq_dic, max_item)
        trainer = Trainer(model, train_dataloader, None, test_dataset, args, logger, is_cache)
        if args.do_eval:
            if args.load_model is None:
                logger.info(f"No model input!")
                exit(0)
            else:
                args.checkpoint_path = os.path.join(args.output_dir, args.load_model + '.pt')
                trainer.load(args.checkpoint_path)
                logger.info(f"Load model from {args.checkpoint_path} for test!")
                scores, result_info = trainer.test(0)

        else:
            early_stopping = EarlyStopping(args.checkpoint_path, logger=logger, patience=args.patience, verbose=True)
            for epoch in range(args.epochs):
                trainer.train(epoch)
                scores, _ = trainer.valid(epoch)
                # evaluate on MRR
                early_stopping(np.array(scores[-1:]), trainer.model)
                if early_stopping.early_stop:
                    logger.info("Early stopping")
                    break

            logger.info("---------------Test Score---------------")
            trainer.model.load_state_dict(torch.load(args.checkpoint_path))
            scores, result_info = trainer.test(0)
    else:
        cache_main_model = get_cache_world_model(args)
        cache_main_model.share_memory()
        train_dataset.set_explore_model(cache_main_model)
        # eval_dataset.set_explore_model(cache_main_model)
        trainer = Trainer(cache_main_model, train_dataloader, None, test_dataset, args, logger, is_cache)
        if args.do_eval:
            trainer.eval_rl()
        elif args.is_replace_imitate:
            run_step = 0
            for epoch in range(args.epochs):
                run_step = trainer.train_rl_imitate_belady(epoch, run_step)
        else:
            run_step = 0
            for epoch in range(args.epochs):
                run_step = trainer.train_rl(epoch, run_step)

    logger.info(args.train_name)
    # logger.info(result_info)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    
    from pyinstrument import Profiler
    
    try:
        with Profiler(interval=0.1) as profiler:
            tr = tracker.SummaryTracker()
            main()
    finally:
        tr.print_diff()
        profiler.print(open(os.path.join(
            os.path.dirname(__file__),
            "profile.txt"
        ), 'w'))
