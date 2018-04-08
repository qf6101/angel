/*
 * Tencent is pleased to support the open source community by making Angel available.
 *
 * Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 */

package io.github.qfeng.angel.apps.featurecount

import com.tencent.angel.ml.conf.MLConf._
import com.tencent.angel.ml.feature.LabeledData
import com.tencent.angel.ml.math.vector.{SparseDoubleSortedVector, SparseIntVector, TIntVector}
import com.tencent.angel.ml.task.TrainTask
import com.tencent.angel.worker.task.TaskContext
import org.apache.commons.logging.{Log, LogFactory}
import org.apache.hadoop.io.{LongWritable, Text}

/**
  * Feature Count task
  *
  * @param ctx taskContext of task
  */
class FeaCountTask(val ctx: TaskContext) extends TrainTask[LongWritable, Text](ctx) {
  val LOG: Log = LogFactory.getLog(classOf[FeaCountTask])
  val feaNum: Int = conf.getInt(ML_FEATURE_NUM, DEFAULT_ML_FEATURE_NUM)
  val dataParser = new VWDataParser(feaNum)


  /**
    * Parse input text as labeled data, X is the feature weight vector, Y is label.
    */
  override
  def parse(key: LongWritable, value: Text): LabeledData = {
    dataParser.parse(value.toString)
  }

  /**
    * Train a LR model iteratively
    *
    * @param ctx context of this Task
    */
  override
  def train(ctx: TaskContext): Unit = {
    val model = new FeaCountModel(conf, ctx)

    taskDataBlock.resetReadIndex()
    for (i <- 0 until taskDataBlock.size) {
      val data = taskDataBlock.read()
      val x = data.getX.asInstanceOf[SparseDoubleSortedVector]
      val countVec = new SparseIntVector(x.size(), x.getIndices, x.values.map(_=>1))
      countVec.setRowId(0)
      model.feaCounter.increment(countVec)
      model.feaCounter.syncClock()
    }

    ctx.incEpoch()
  }
}
