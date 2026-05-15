# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastgen.methods.model import FastGenModel as FastGenModel

from fastgen.methods.distribution_matching.dmd2 import DMD2Model as DMD2Model
from fastgen.methods.distribution_matching.ladd import LADDModel as LADDModel
from fastgen.methods.distribution_matching.f_distill import FdistillModel as FdistillModel

from fastgen.methods.distribution_matching.causvid import CausVidModel as CausVidModel
from fastgen.methods.distribution_matching.self_forcing import SelfForcingModel as SelfForcingModel
from fastgen.methods.distribution_matching.anyflow import AnyFlowModel as AnyFlowModel

from fastgen.methods.consistency_model.CM import CMModel as CMModel
from fastgen.methods.consistency_model.TCM import TCMModel as TCMModel
from fastgen.methods.consistency_model.sCM import SCMModel as SCMModel
from fastgen.methods.consistency_model.mean_flow import MeanFlowModel as MeanFlowModel

from fastgen.methods.fine_tuning.sft import SFTModel as SFTModel
from fastgen.methods.fine_tuning.sft import CausalSFTModel as CausalSFTModel

from fastgen.methods.knowledge_distillation.KD import KDModel as KDModel
from fastgen.methods.knowledge_distillation.KD import CausalKDModel as CausalKDModel
