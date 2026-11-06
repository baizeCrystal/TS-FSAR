import numpy as np
import pprint
import torch
import math
import torch.nn as nn
import utils.optimizer as optim
import utils.logging as logging
import utils.tools as tl
from utils.tools import TrainMeter
from trainers.model  import build_model
from datasets.builder import build_loader, shuffle_dataset
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from utils.functions import cosine_lr, extract_class_indices,build_embeddings
from tqdm import tqdm
import  math
from trainers.ts_fsar import TS_FSAR
import utils.logging as logging
from utils.config import Config
import os
import sys
import time
import json


scaler = GradScaler()
logger = logging.get_logger(__name__)

def train_epoch(
    train_loader,val_loader, model, train_embeddings, test_embeddings, optimizer, lr_scheduler, train_meter, cfg
):
    model.train()
    train_meter.iter_tic()
    total_loss = 0.
    best_acc = -float('inf')
    best_iter = None
    for cur_iter, task_dict in enumerate(train_loader):
        if cur_iter >= cfg.TRAIN.NUM_TRAIN_TASKS:
                break
        for key in task_dict.keys():
            task_dict[key] = task_dict[key][0].cuda(non_blocking=True)
        context_imgs = task_dict['support_set'].squeeze(0).cuda()
        context_labels = task_dict['support_labels'].squeeze(0).long()
        target_imgs = task_dict['target_set'].squeeze(0).cuda()
        target_labels = task_dict['target_labels'].squeeze(0).long()
        context_real_labels = task_dict['real_support_labels'].squeeze(0).long().cuda()
        target_real_labels = task_dict['real_target_labels'].squeeze(0).long().cuda()
        real_labels = torch.cat((context_real_labels, target_real_labels), dim=0)
        imgs = torch.cat((context_imgs, target_imgs), dim=0)#[240,3,224,224]
        with autocast():
            output_dict = model(
                images=imgs,
                context_labels=context_labels,
                target_labels=target_labels,
                real_labels=real_labels,        
                class_embeddings=train_embeddings
            )
            loss = output_dict['loss']
        scaler.scale(loss).backward()
        if ((cur_iter + 1) >= int(getattr(cfg.TRAIN, 'EVAL_START_EPISODE', 1))
                and (cur_iter + 1) % cfg.TRAIN.VAL_FRE_ITER == 0):
            model.eval()
            print("Validating ...")
            acc, _ = test(val_loader, model, test_embeddings, cfg)
            model.train()
            if not math.isfinite(acc):
                raise RuntimeError(f"Non-finite validation accuracy at iter {cur_iter}: {acc}")
            if acc > best_acc:
                best_acc = acc
                best_iter = cur_iter
                best_path = os.path.join(cfg.OUTPUT_DIR_SAVE, 'model_best.pth')
                torch.save(model.state_dict(), best_path + '.tmp')
                os.replace(best_path + '.tmp', best_path)
                logger.info(
                    f"New Best Acc at iter {cur_iter}: {best_acc:.2f}% ")
            logger.info(
                "[VAL] Iter %d: Acc=%.2f%% | Best Acc=%.2f%% at episode %d (iter %d)",
                cur_iter, acc, best_acc, best_iter + 1, best_iter)
            best_result = {
                'best_accuracy_percent': float(best_acc),
                'best_iteration': best_iter,
                'best_episode': best_iter + 1,
                'latest_accuracy_percent': float(acc),
                'latest_iteration': cur_iter,
                'checkpoint': os.path.join(cfg.OUTPUT_DIR_SAVE, 'model_best.pth'),
            }
            result_path = os.path.join(cfg.OUTPUT_DIR_SAVE, 'best_result.json')
            with open(result_path + '.tmp', 'w') as stream:
                json.dump(best_result, stream, indent=2)
                stream.write('\n')
            os.replace(result_path + '.tmp', result_path)
        if ((cur_iter + 1) % cfg.TRAIN.BATCH_SIZE_PER_TASK == 0):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        if math.isnan(loss.item()):
            raise RuntimeError("ERROR: Got NaN losses")
        total_loss += loss.item()
        train_meter.update_stats(
            top1_err=100.0 - (output_dict['acc'] * 100.0),
            loss=loss.item(),
            lr=optimizer.param_groups[0]["lr"],
            mb_size=1
        )
        train_meter.log_iter_stats(cur_iter)
        train_meter.iter_tic()
        lr_scheduler(cur_iter)
    train_meter.log_epoch_stats()
    if best_iter is not None:
        logger.info(
            "[TRAIN_SUMMARY] Best validation accuracy=%.2f%% at episode %d (iter %d) | checkpoint=%s",
            best_acc, best_iter + 1, best_iter,
            os.path.join(cfg.OUTPUT_DIR_SAVE, 'model_best.pth'))
    train_meter.reset()

def test(val_loader, model, class_embeddings, cfg):
    model.eval()
    test_accs = []
    test_losses = []
    num_episodes = len(val_loader)
    log_period = max(1, int(getattr(cfg.TEST, "PROGRESS_LOG_PERIOD", 1000)))
    started_at = time.perf_counter()
    progress_bar = tqdm(val_loader, total=num_episodes, desc=f"Processing {cfg.TEST.DATASET}",
                        disable=not sys.stderr.isatty(), mininterval=5.0)
    for episode_index, task_dict in enumerate(progress_bar, start=1):
        context_imgs = task_dict['support_set'].squeeze(0).cuda()
        context_labels = task_dict['support_labels'].squeeze(0).long().cuda()
        context_real_labels = task_dict['real_support_labels'].squeeze(0).long().cuda()
        target_imgs = task_dict['target_set'].squeeze(0).cuda()
        target_labels = task_dict['target_labels'].squeeze(0).long().cuda()
        unique_labels = torch.unique(context_labels)
        with torch.no_grad():
            text_embedding = [torch.mean(torch.index_select(class_embeddings[context_real_labels, :], 0, extract_class_indices(context_labels, c)), dim=0) for c in unique_labels]
            text_embedding = torch.stack(text_embedding)
            images = torch.cat((context_imgs, target_imgs), dim=0)
            output_dict = model(
                images=images,
                context_labels=context_labels,
                target_labels=target_labels,
                real_labels=None,             
                class_embeddings=text_embedding
            )

            acc = output_dict['acc']
            loss = output_dict['loss']

            test_accs.append(acc)
            test_losses.append(loss.item())
        if episode_index % log_period == 0 or episode_index == num_episodes:
            elapsed = time.perf_counter() - started_at
            rate = episode_index / max(elapsed, 1e-9)
            logger.info("[EVAL_PROGRESS] %d/%d episodes | %.2f episodes/s | elapsed %.1fs | ETA %.1fs",
                        episode_index, num_episodes, rate, elapsed,
                        (num_episodes - episode_index) / rate)

    avg_acc = np.mean(test_accs) * 100
    avg_loss = np.mean(test_losses)
    print(f"{cfg.TEST.DATASET} Result: Acc {avg_acc:.2f}%, Loss {avg_loss:.4f}")
    
    return avg_acc,avg_loss


def main(cfg):
    """
    Args:
        cfg (Config): The global config object.
    """
    if cfg.TRAIN.ENABLE:
        seed = cfg.TRAIN.RANDOM_SEED
        current_state = "TRAIN" #current_state
        log_file = cfg.TRAIN.LOG_FILE
    elif cfg.TEST.ENABLE:
        seed = cfg.TRAIN.RANDOM_SEED
        current_state = "TEST"
        log_file = cfg.TEST.LOG_FILE
    print(f"Current Execution Mode: {current_state}")
    print(f"Setting Random Seed to: {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.setup_logging(cfg, log_file)
    logger = logging.get_logger(__name__)

    if cfg.LOG_CONFIG_INFO:
        logger.info("Running with config:")
        logger.info(pprint.pformat(cfg))
    image_clip, text_clip, tokenizer = build_model(cfg)
    model = TS_FSAR(cfg, image_clip).cuda()
    if cfg.TRAIN.ENABLE:
        logger.info(">>>>>>>> Starting Training >>>>>>>>")
        train_loader = build_loader(cfg, "train")
        val_loader = build_loader(cfg, "test")
        optimizer = optim.construct_optimizer(model, cfg)
        lr_scheduler = cosine_lr(optimizer, 0, cfg.TRAIN.NUM_TRAIN_TASKS)
        text_embeddings = build_embeddings(cfg.TRAIN.CLASS_NAME, cfg.TRAIN.LLM_train_json, text_clip, tokenizer)
        text_embeddings_val = build_embeddings(cfg.TEST.CLASS_NAME, cfg.TEST.LLM_test_json, text_clip, tokenizer)
        train_meter = TrainMeter(min(len(train_loader), cfg.TRAIN.NUM_TRAIN_TASKS), cfg)
        logger.info("Start epoch: {}".format(cfg.TRAIN.START_EPOCH + 1))
        if cfg.LOG_MODEL_INFO:
            tl.log_model_info(model, cfg, use_train_input=True)
        cur_epoch = 0
        shuffle_dataset(train_loader, cur_epoch)
        train_epoch(
            train_loader,val_loader, model, text_embeddings,text_embeddings_val,
            optimizer, lr_scheduler, train_meter, cfg
        )
        logger.info("Training Finished.")
    if cfg.TEST.ENABLE:
        logger.info(">>>>>>>> Starting Testing >>>>>>>>")
        if not cfg.TRAIN.ENABLE:
            if hasattr(cfg.TEST, "LOAD_PRETRAIN") and cfg.TEST.LOAD_PRETRAIN:
                logger.info(f"Loading pretrained model for testing from {cfg.TEST.PRETRAIN_MODEL_PATH}")
                pretrained_dict = torch.load(cfg.TEST.PRETRAIN_MODEL_PATH)
                model.load_state_dict(pretrained_dict, strict=True)
                logger.info("Pre-trained weights loaded successfully.")
            else:
                logger.warning("WARNING: Test enabled but no pretrain path provided! Testing with random weights.")
        else:
            logger.info("Continuing testing with the trained best model...")
            best_model_path = cfg.OUTPUT_DIR_SAVE + '/model_best.pth'
            try:
                model.load_state_dict(torch.load(best_model_path))
                logger.info("Loaded model_best.pth")
            except Exception as e:
                logger.warning(f"Could not load best model ({e}), using current model state.")
        val_loader = build_loader(cfg, "test")
        text_embeddings = build_embeddings(cfg.TEST.CLASS_NAME, cfg.TEST.LLM_test_json, text_clip, tokenizer)
        acc, loss = test(val_loader, model, text_embeddings, cfg)
        logger.info(
            f"Final Test Accuracy: "
            f"overall={acc:.2f}%, "
            f"loss={loss:.2f}"
        )
if __name__ == "__main__":
    cfg = Config(load=True)
    main(cfg)
    print("Finish running with config: {}".format(cfg.args.cfg_file))


  







