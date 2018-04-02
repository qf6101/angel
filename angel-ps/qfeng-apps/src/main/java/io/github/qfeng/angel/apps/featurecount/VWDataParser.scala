package io.github.qfeng.angel.apps.featurecount

import com.tencent.angel.ml.feature.LabeledData
import com.tencent.angel.ml.utils.DataParser
import com.tencent.angel.ml.math.vector.SparseDoubleSortedVector

case class VWDataParser(val feaNum: Int) extends DataParser {
  override def parse(value: String) = {
    val splits = value.split("\\|")
    val y = if (splits(0).trim() == "-1") 0.0 else 1.0

    val (feaIndices, feaValues) = splits(1).trim().split(" ").map { str =>
      val elem = str.split(":")
      if(elem.length == 1) (elem(0).toInt, 1.0) else (elem(0).toInt, elem(1).toDouble)
    }.toMap.toArray.sortBy(_._1).unzip

    val x = new SparseDoubleSortedVector(feaNum, feaIndices, feaValues)
    new LabeledData(x, y)
  }
}
