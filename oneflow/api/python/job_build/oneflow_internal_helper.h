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
#ifndef ONEFLOW_API_PYTHON_JOB_BUILD_JOB_BUILD_AND_INFER_HELPER_H_
#define ONEFLOW_API_PYTHON_JOB_BUILD_JOB_BUILD_AND_INFER_HELPER_H_

#include "oneflow/core/job/job_build_and_infer_ctx.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/job.pb.h
#include "oneflow/core/job/job.cfg.h
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/job/placement.cfg.h"

namespace oneflow {

namespace {

bool EagerExecutionEnabled() { return *Global<bool, EagerExecution>::Get(); }

Maybe<JobBuildAndInferCtxMgr*> GlobalJobBuildAndInferCtxMgr() {
  if (EagerExecutionEnabled()) {
    return JUST(GlobalMaybe<EagerJobBuildAndInferCtxMgr>());
  } else {
    return JUST(GlobalMaybe<LazyJobBuildAndInferCtxMgr>());
  }
}

Maybe<JobBuildAndInferCtx*> GetJobBuildAndInferCtx(const std::string& job_name) {
  auto* mgr = JUST(GlobalJobBuildAndInferCtxMgr());
  return mgr->FindJobBuildAndInferCtx(job_name);
}

Maybe<JobBuildAndInferCtx*> GetCurInferCtx() {
  auto* mgr = JUST(GlobalJobBuildAndInferCtxMgr());
  return mgr->FindJobBuildAndInferCtx(*JUST(mgr->GetCurrentJobName()));
}

} // namespace


Maybe<void> CurJobBuildAndInferCtx_SetJobConf(const std::shared_ptr<cfg::JobConfigProto>& cfg_job_conf) {
  JobConfigProto job_conf;
  cfg_job_conf->ToProto(&job_conf);
  return JUST(GetCurInferCtx())->SetJobConf(job_conf);
}

Maybe<std::string> JobBuildAndInferCtx_MirroredBlobGetSerializedParallelConfFromProducerView(
    const std::string& job_name, const std::string& lbn) {
  auto* ctx = JUST(GetJobBuildAndInferCtx(job_name));
  return PbMessage2TxtString(
      JUST(ctx->MirroredBlobGetParallelDescFromProducerView(lbn))->parallel_conf());
}

Maybe<std::string> GetSerializedMachineId2DeviceIdListOFRecord(
    const std::shared_ptr<cfg::ParallelConf>& cfg_parallel_conf) {
  ParallelConf parallel_conf;
  cfg_parallel_conf->ToProto(&parallel_conf);
  return PbMessage2TxtString(*JUST(ParseMachineAndDeviceIdList(parallel_conf)));
}

} // namespace oneflow
#endif  // ONEFLOW_API_PYTHON_JOB_BUILD_JOB_BUILD_AND_INFER_HELPER_H_
