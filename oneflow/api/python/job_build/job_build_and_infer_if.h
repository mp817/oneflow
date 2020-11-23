/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include <utility>
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/job_build_and_infer_ctx.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/job_conf.cfg.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/job/placement.cfg.h"
#include "oneflow/api/python/job_build/job_build_and_infer_helper.h"


std::shared_ptr<oneflow::cfg::ErrorProto> CurJobBuildAndInferCtx_SetJobConf(const std::string& serialized_job_conf) {
  return oneflow::CurJobBuildAndInferCtx_SetJobConf(serialized_job_conf).GetDataAndErrorProto();
}

std::pair<std::string, std::shared_ptr<oneflow::cfg::ErrorProto>> JobBuildAndInferCtx_MirroredBlobGetSerializedParallelConfFromProducerView(
    const std::string& job_name, const std::string& lbn) {
  return oneflow::JobBuildAndInferCtx_MirroredBlobGetSerializedParallelConfFromProducerView(job_name, lbn).GetDataAndErrorProto(std::string(""));
}

std::pair<std::string, std::shared_ptr<oneflow::cfg::ErrorProto>> GetMachine2DeviceIdListOFRecordFromParallelConf(const std::string& parallel_conf) {
  return oneflow::GetSerializedMachineId2DeviceIdListOFRecord(parallel_conf)
      .GetDataAndErrorProto(std::string(""));
}

