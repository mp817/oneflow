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

#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/job_build_and_infer_ctx.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/job_conf.cfg.h"
#include "oneflow/core/job/job_conf.pb.h"
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

}  // namespace

Maybe<void> JobBuildAndInferCtx_Open(const std::string& job_name) {
  auto* mgr = JUST(GlobalJobBuildAndInferCtxMgr());
  return mgr->OpenJobBuildAndInferCtx(job_name);
}

Maybe<std::string> JobBuildAndInferCtx_GetCurrentJobName() {
  auto* mgr = JUST(GlobalJobBuildAndInferCtxMgr());
  return mgr->GetCurrentJobName();
}

Maybe<void> JobBuildAndInferCtx_Close() {
  auto* mgr = JUST(GlobalJobBuildAndInferCtxMgr());
  JUST(mgr->CloseCurrentJobBuildAndInferCtx());
  return Maybe<void>::Ok();
}

Maybe<void> CurJobBuildAndInferCtx_CheckJob() { return JUST(GetCurInferCtx())->CheckJob(); }

Maybe<void> CurJobBuildAndInferCtx_SetJobConf(const std::string& serialized_job_conf) {
  // parse
  JobConfigProto job_conf;
  CHECK_OR_RETURN(TxtString2PbMessage(serialized_job_conf, &job_conf)) << "job conf parse failed";
  return JUST(GetCurInferCtx())->SetJobConf(job_conf);
}

Maybe<void> CurJobBuildAndInferCtx_SetTrainConf(const std::string& train_conf_str) {
  TrainConf train_conf;
  CHECK_OR_RETURN(TxtString2PbMessage(train_conf_str, &train_conf)) << "train conf parse failed";
  return JUST(GetCurInferCtx())->SetTrainConf(train_conf);
}

Maybe<void> CurJobBuildAndInferCtx_Complete() { return JUST(GetCurInferCtx())->Complete(); }

Maybe<bool> CurJobBuildAndInferCtx_HasJobConf() { return JUST(GetCurInferCtx())->HasJobConf(); }

Maybe<std::string> CurJobBuildAndInferCtx_AddAndInferMirroredOp(const std::string& op_conf_str) {
  OperatorConf op_conf;
  CHECK_OR_RETURN(TxtString2PbMessage(op_conf_str, &op_conf)) << "operator conf parse failed";
  auto* ctx = JUST(GetCurInferCtx());
  const auto& op_attribute = JUST(ctx->AddAndInferMirroredOp(op_conf));
  return PbMessage2TxtString(*op_attribute);
}

Maybe<std::string> CurJobBuildAndInferCtx_AddAndInferConsistentOp(const std::string& op_conf_str) {
  OperatorConf op_conf;
  CHECK_OR_RETURN(TxtString2PbMessage(op_conf_str, &op_conf)) << "operator conf parse failed";
  auto* ctx = JUST(GetCurInferCtx());
  const auto& op_attribute = JUST(ctx->AddAndInferConsistentOp(op_conf));
  return PbMessage2TxtString(*op_attribute);
}

Maybe<std::string> JobBuildAndInferCtx_MirroredBlobGetSerializedParallelConfFromProducerView(
    const std::string& job_name, const std::string& lbn) {
  auto* ctx = JUST(GetJobBuildAndInferCtx(job_name));
  return PbMessage2TxtString(
      JUST(ctx->MirroredBlobGetParallelDescFromProducerView(lbn))->parallel_conf());
}

Maybe<std::string> GetSerializedMachineId2DeviceIdListOFRecord(
    const std::string& parallel_conf_str) {
  ParallelConf parallel_conf;
  CHECK_OR_RETURN(TxtString2PbMessage(parallel_conf_str, &parallel_conf))
      << "parallel conf parse failed";
  return PbMessage2TxtString(*JUST(ParseMachineAndDeviceIdList(parallel_conf)));
}

}  // namespace oneflow
#endif  // ONEFLOW_API_PYTHON_JOB_BUILD_JOB_BUILD_AND_INFER_HELPER_H_
