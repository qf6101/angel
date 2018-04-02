package io.github.qfeng.angel.apps.featurecount

import com.tencent.angel.ml.conf.MLConf
import com.tencent.angel.ml.feature.LabeledData
import com.tencent.angel.ml.model.{MLModel, PSModel}
import com.tencent.angel.ml.predict.PredictResult
import com.tencent.angel.ml.matrix.RowType
import com.tencent.angel.worker.storage.DataBlock
import com.tencent.angel.worker.task.TaskContext
import org.apache.hadoop.conf.Configuration

class FeaCountModel(conf: Configuration, _ctx: TaskContext = null) extends MLModel(conf, _ctx){
  val N: Int = conf.getInt(MLConf.ML_FEATURE_NUM, MLConf.DEFAULT_ML_FEATURE_NUM)

  val feaCounter = PSModel("feature_counts", 1, N).setRowType(RowType.T_INT_SPARSE).setAverage(true)
  addPSModel(feaCounter)

  setSavePath(conf)
  setLoadPath(conf)

  /**
    * Predict use the PSModels and predict data
    *
    * @param storage predict data
    * @return predict result
    */
  override def predict(storage: DataBlock[LabeledData]): DataBlock[PredictResult] = ???
}