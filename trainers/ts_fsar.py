import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from utils.functions import cos_sim, extract_class_indices, DistillKL
from trainers.model import LSN
from trainers.frame_tsm import FrameMatchingGenerator
from trainers.support_weighting import DiscriminativeSupportAggregation
from trainers.tsm_residual import TSMResidual, support_condition
import utils.logging as logging

logger = logging.get_logger(__name__)

class Alpha_DC(nn.Module):
    def __init__(self, alpha = 0.4):
        super().__init__()
        self.alpha = alpha

    def _triuvec(self, x):
        """
        Extracts the upper triangular part of the matrix as a flattened vector.
        This method is sourced from https://github.com/Fei-Long121/DeepBDC.
        """
        batchSize, dim, dim = x.shape
        r = x.reshape(batchSize, dim * dim)
        I = torch.ones(dim, dim).triu().reshape(dim * dim)
        index = I.nonzero(as_tuple=False)
        y = r[:, index.squeeze()]
        return y

    def _alpha_dcov(self, x):
        """
        Computes the alpha-Distance Covariance matrix (alpha-D Matrix).
        This method is modified from https://github.com/Fei-Long121/DeepBDC.
        Modifications include custom handling for the alpha hyperparameter and stability adjustments.
        """
        batchSize, dim, M = x.data.shape
        I = torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(x.dtype)
        I_M = torch.ones(batchSize, dim, dim, device=x.device).type(x.dtype)
        x_pow2 = x.bmm(x.transpose(1, 2) * 1./ (2*M))
        dcov = I_M.bmm(x_pow2 * I) + (x_pow2 * I).bmm(I_M) - 2 * x_pow2
        if torch.isnan(dcov).any().item():
            print("dcov is nan")
        dcov = torch.clamp(dcov, min=0.0)
        dcov = torch.pow(dcov + 1e-5, self.alpha)
        d1 = dcov.bmm(I_M * 1./ dim)
        d2 = (I_M * 1./ dim ).bmm(dcov)
        d3 = (I_M * 1./ dim ).bmm(dcov).bmm(I_M * 1./ dim)
        out = dcov - d1 - d2 + d3
        return out

    def forward(self, features):
        repre = self._alpha_dcov(features)
        repre = self._triuvec(repre)
        return repre

class TS_DCM(nn.Module):
    def __init__(self, cfg, hidden_dim):
        super().__init__()
        self.cfg = cfg
        self.alpha_dc = Alpha_DC(cfg.TRAIN.DC_ALPHA)
        self.generator = nn.Linear(hidden_dim, cfg.DATA.NUM_INPUT_FRAMES * cfg.DATA.NUM_INPUT_FRAMES, bias=False)
        self.support_aggregation_mode = getattr(cfg, 'TSM_SUPPORT_AGGREGATION', 'mean')
        if self.support_aggregation_mode not in ('mean','query_softmax','team_diff_query_softmax'):
            raise ValueError('Unknown TSM_SUPPORT_AGGREGATION: '+self.support_aggregation_mode)
        if self.support_aggregation_mode != 'mean':
            self.support_aggregator = DiscriminativeSupportAggregation(cfg)

        self.tsm_variant = getattr(cfg, "TSM_VARIANT", "original")
        if self.tsm_variant not in ("original", "frame", "ematch_frame"):
            raise ValueError("Unknown TSM_VARIANT: " + self.tsm_variant)
        if self.tsm_variant != "original":
            # Keep the original RNG stream for all existing modules and dropout.
            with torch.random.fork_rng(devices=[]):
                self.frame_generator = FrameMatchingGenerator(cfg, hidden_dim)
        residual = getattr(cfg, 'TSM_RESIDUAL', None)
        self.residual_enabled = bool(getattr(residual, 'ENABLE', False))
        if self.residual_enabled:
            if self.tsm_variant != 'original' or self.support_aggregation_mode != 'mean':
                raise ValueError('Residual ablation must start from original TSM')
            if float(getattr(residual, 'SCALE', 1.0)) != 1.0:
                raise ValueError('This experiment fixes residual lambda=1')
            # Construction is CPU-only and restores the existing RNG stream.
            # "generator" also puts these parameters in the ORIGINAL optimizer's
            # generator group, with its unchanged LR/weight-decay/schedule.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(getattr(residual, 'INIT_SEED', 0))
                self.residual_generator = TSMResidual(residual.MODE, residual.SUPPORT_CONDITION)
        self.residual_observer = None

    def forward(self, lsn_output, context_labels, way, shot, query_per_class):
        lsn_cls = lsn_output[:, 0]       
        lsn_patch = lsn_output[:, 1:]    
        weight = F.cosine_similarity(lsn_cls.unsqueeze(1), lsn_patch, dim=-1)
        weight = weight.view(*weight.size()[:], 1)
        weighted_patch = weight * lsn_patch
        adc_repre = self.alpha_dc(weighted_patch.transpose(1, 2))# alpha distace correlation
        num_support = way * shot * self.cfg.DATA.NUM_INPUT_FRAMES
        context_features = adc_repre[:num_support]
        target_features = adc_repre[num_support:]
        support_cls = lsn_cls[:num_support]
        query_cls = lsn_cls[num_support:]
        task_query = query_cls.reshape(-1, self.cfg.DATA.NUM_INPUT_FRAMES, *query_cls.size()[1:]).mean(1) 
        task_query = task_query.reshape(-1, 1, task_query.size()[-1]).repeat(1, way, 1) 
        
        task_support = support_cls.reshape(way * shot, self.cfg.DATA.NUM_INPUT_FRAMES, -1).mean(1).mean(0)
        task_support = task_support.reshape(1, 1, -1).repeat(way * query_per_class, way, 1)

        qs_prototype = task_support + task_query #query-specific task prototype
        if self.support_aggregation_mode != 'mean':
            video_support = support_cls.reshape(way*shot,self.cfg.DATA.NUM_INPUT_FRAMES,-1).mean(1)
            prototype = self.support_aggregator(task_query[:,0], video_support, context_labels)
            qs_prototype = prototype[:,None,:].expand(-1,way,-1)
        tsm_matrix = self.generator(qs_prototype) # task-specific matching matrix

        unique_labels = torch.unique(context_labels)
        sample_context_features = context_features.reshape(way * shot, self.cfg.DATA.NUM_INPUT_FRAMES, -1)
        context_features = context_features.reshape(way * shot, self.cfg.DATA.NUM_INPUT_FRAMES,  -1 )
        context_features = [torch.mean(torch.index_select(context_features, 0, extract_class_indices(context_labels, c)), dim=0) for c in unique_labels]
            
        z_proto = torch.stack(context_features)
        z_proto = rearrange(z_proto, 'w f d -> (w f) d')
        z_query = target_features
        ad_matrix = cos_sim(z_query, z_proto) # inter-frame alpha-Distance Correlation/alpha-D matrix
        ad_matrix = rearrange(ad_matrix, '(tb ts) (sb ss) -> tb sb ts ss', tb=way*query_per_class, sb = way)
        score = (ad_matrix.reshape(way * query_per_class, way, -1) * tsm_matrix).sum(-1)
        if self.residual_enabled:
            frames = self.cfg.DATA.NUM_INPUT_FRAMES
            base_weight = tsm_matrix.reshape(-1, way, frames, frames)
            bc = None
            if self.residual_generator.conditioned:
                bc = support_condition(sample_context_features, context_labels, unique_labels)
            delta = self.residual_generator(ad_matrix, base_weight, bc)
            changed = (ad_matrix.reshape(way * query_per_class, way, -1) *
                       (base_weight + delta).reshape(way * query_per_class, way, -1)).sum(-1)
            if self.residual_observer is not None:
                self.residual_observer(ad_matrix, base_weight, delta, score, changed, self.training)
            score = changed
        if self.tsm_variant != "original":
            frames = self.cfg.DATA.NUM_INPUT_FRAMES
            query_frames = query_cls.reshape(-1, frames, query_cls.shape[-1])
            support_frames = support_cls.reshape(way * shot, frames, support_cls.shape[-1])
            initial = None
            if self.tsm_variant == "ematch_frame":
                # Sample-level selection only; never alter the class-level M_corr above.
                sample_corr = cos_sim(target_features, sample_context_features.reshape(way * shot * frames, -1))
                sample_corr = rearrange(sample_corr, '(q t) (s u) -> q s t u', q=way*query_per_class, s=way*shot)
                initial = (sample_corr * tsm_matrix[:, :1].reshape(-1, 1, frames, frames)).sum((-1, -2))
                if self.frame_generator.statistics is not None:
                    self._initial_for_statistics = initial.detach()
            delta = self.frame_generator(query_frames, support_frames, initial)
            # Original axes are [query frame, support frame]. Average weights over
            # same-class samples, keeping TS-FSAR's original feature-prototype protocol.
            delta_class = torch.stack([
                delta.index_select(1, extract_class_indices(context_labels, c)).mean(1)
                for c in unique_labels
            ], dim=1)
            final = tsm_matrix.reshape(-1, way, frames, frames) + self.frame_generator.residual_scale * delta_class
            score = (ad_matrix * final).sum((-1, -2))
        return score, adc_repre, lsn_cls

class TS_FSAR(nn.Module):
    def __init__(self, cfg, image_clip):
        super().__init__()
        self.cfg = cfg
        print("Building LSN inside TS_FSAR...")
        self.freeze_backbone(image_clip)
        self.model = LSN(image_clip, cfg)
        dim_repre = int(cfg.LSN.OUTPUT_DIM * (cfg.LSN.OUTPUT_DIM+1) / 2)
        setattr(self.model, 'Linear', nn.Linear(dim_repre, cfg.TRAIN.NUM_CLASSES, bias=False))
        setattr(self.model, 'dropout', nn.Dropout(cfg.LSN.DROPOUT))
        embed_dim = cfg.LSN.EMBED_DIM
        self.ts_dcm = TS_DCM(cfg, embed_dim)
        self.kl = DistillKL(cfg.TRAIN.KD_T)
        self.ce = nn.CrossEntropyLoss()

    def freeze_backbone(self, model):
        for p in model.parameters():
            p.requires_grad = False
        
    def forward(self, 
                images,           
                context_labels,   
                target_labels,    
                real_labels=None, 
                class_embeddings=None):
        _, c, h, w = images.size()
        x = images.view(-1, c, h, w)
        lsn_features, clip_cls = self.model(x)
        way = self.cfg.TRAIN.WAY
        shot = self.cfg.TRAIN.SHOT
        query_per_class = self.cfg.TRAIN.QUERY_PER_CLASS if self.model.training else self.cfg.TRAIN.QUERY_PER_CLASS_TEST
        score, adc_repre, lsn_cls = self.ts_dcm(
            lsn_features, context_labels,
            way, shot, query_per_class
        )
        if (self.ts_dcm.tsm_variant == 'ematch_frame'
                and self.ts_dcm.frame_generator.statistics is not None):
            self.ts_dcm.frame_generator.statistics.update(
                self.ts_dcm._initial_for_statistics, context_labels, target_labels, self.training)
        if self.model.training:
            logits_vision = F.log_softmax(score, dim=1)
            loss_vision = F.nll_loss(logits_vision, target_labels.type(torch.long)) #TS-DCM
            acc_vision = torch.eq(score.argmax(1), target_labels).float().mean()

            feat_adapted = adc_repre.reshape(-1, self.cfg.DATA.NUM_INPUT_FRAMES, adc_repre.size()[-1]).mean(1) # Guided LSN from Adapted CLIP
            logits_lsn_adc = self.model.Linear(self.model.dropout(feat_adapted))
            loss_gc = self.ce(logits_lsn_adc, real_labels) # GLAC-CE
    
            feat_clip = clip_cls.reshape(-1, self.cfg.DATA.NUM_INPUT_FRAMES, clip_cls.size()[-1])
            feat_clip = self.model.Adapter(feat_clip).mean(1)
            feat_clip_norm = feat_clip / feat_clip.norm(dim=-1, keepdim=True)
            logits_v2t_clip = feat_clip_norm @ class_embeddings.t() * self.cfg.TEMP_CLIP
            loss_text_clip = self.ce(logits_v2t_clip, real_labels) # GLAC-CE
            
            feat_lsn = lsn_cls.reshape(-1, self.cfg.DATA.NUM_INPUT_FRAMES, lsn_cls.size()[-1])
            feat_lsn = self.model.up_linear(self.model.up_ln(feat_lsn)).mean(1)
            feat_lsn_norm = feat_lsn / feat_lsn.norm(dim=-1, keepdim=True)
            logits_v2t_lsn = feat_lsn_norm @ class_embeddings.t() * self.cfg.TEMP_SIDE
            loss_text_lsn = self.ce(logits_v2t_lsn, real_labels) #LSN
            loss_gk = self.kl(logits_lsn_adc, logits_v2t_clip) # GLAC-KL
            glac = getattr(self.cfg, 'GLAC', None)
            enabled = getattr(glac, 'ENABLE', True)
            use_kl = enabled and getattr(glac, 'ENABLE_KL', True)
            use_student_ce = enabled and getattr(glac, 'ENABLE_STUDENT_CE', True)
            use_teacher_ce = enabled and getattr(glac, 'ENABLE_TEACHER_CE', True)
            # Keep the complete original forward and RNG calls. Disabled terms
            # remain measured for logging, but have no path into the total loss.
            loss = (
                self.cfg.TRAIN.VIS_ALPHA * loss_vision +
                self.cfg.TRAIN.CLS_ALPHA * (loss_gc if use_student_ce else 0.) +
                self.cfg.TRAIN.SIDE_CLS_ALPHA * loss_text_lsn +
                self.cfg.TRAIN.KD_ALPHA * ((loss_gk if use_kl else 0.) + (loss_text_clip if use_teacher_ce else 0.))
            )
            if getattr(glac, 'LOG_COMPONENTS', False):
                self.last_loss_components = {
                    'total': loss.detach(), 'ts_dcm': loss_vision.detach(),
                    'lsn_ce': loss_text_lsn.detach(), 'kl': loss_gk.detach(),
                    'student_ce': loss_gc.detach(), 'teacher_ce': loss_text_clip.detach(),
                    'kl_enabled': use_kl, 'student_ce_enabled': use_student_ce,
                    'teacher_ce_enabled': use_teacher_ce,
                }
            if self.ts_dcm.support_aggregation_mode != 'mean':
                self.ts_dcm.support_aggregator.record_accuracy(score.argmax(1),target_labels,self.training)
            return {
                "loss": loss,
                "acc": acc_vision.item()
            }
        else:
            log_vision = F.softmax(score, dim=1)
            num_support = self.cfg.TRAIN.WAY * self.cfg.TRAIN.SHOT * self.cfg.DATA.NUM_INPUT_FRAMES
            feat_clip = clip_cls[num_support:].reshape(-1, self.cfg.DATA.NUM_INPUT_FRAMES, clip_cls.size()[-1])
            feat_clip = self.model.Adapter(feat_clip).mean(1)
            feat_clip_norm = feat_clip / feat_clip.norm(dim=-1, keepdim=True)
            logits_v2t_clip = feat_clip_norm @ class_embeddings.t() * self.cfg.TEMP_CLIP
            log_v2t_clip = F.softmax(logits_v2t_clip, dim=1)

            feat_lsn = lsn_cls[num_support:].reshape(-1, self.cfg.DATA.NUM_INPUT_FRAMES, lsn_cls.size()[-1])
            feat_lsn = self.model.up_linear(self.model.up_ln(feat_lsn)).mean(1)
            feat_lsn_norm = feat_lsn / feat_lsn.norm(dim=-1, keepdim=True)
            logits_v2t_lsn = feat_lsn_norm @ class_embeddings.t() * self.cfg.TEMP_SIDE
            log_v2t_lsn = F.softmax(logits_v2t_lsn, dim=1)
            log = log_vision.pow(self.cfg.TEST.FUSE_VISION) * log_v2t_clip.pow(self.cfg.TEST.FUSE_CLIP) * log_v2t_lsn.pow(self.cfg.TEST.FUSE_SIDE)
            pred = log.argmax(dim=1)
            acc = torch.eq(pred, target_labels).float().mean()
            loss_val = F.nll_loss(torch.log(log + 1e-8), target_labels.type(torch.long))
            if self.ts_dcm.support_aggregation_mode != 'mean':
                self.ts_dcm.support_aggregator.record_accuracy(pred,target_labels,self.training)

            return {
                "loss": loss_val,
                "acc": acc.item()
            }
