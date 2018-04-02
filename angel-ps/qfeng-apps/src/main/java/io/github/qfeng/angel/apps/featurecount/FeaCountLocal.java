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
 */

package io.github.qfeng.angel.apps.featurecount;

import com.tencent.angel.conf.AngelConf;
import com.tencent.angel.ml.conf.MLConf;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.LocalFileSystem;
import org.apache.hadoop.mapreduce.lib.input.CombineTextInputFormat;
import org.apache.log4j.PropertyConfigurator;


public class FeaCountLocal {
  private Configuration conf = new Configuration();

  static {
    PropertyConfigurator.configure("/home/qfeng/projects/angel/dist/target/angel-1.4.0-bin/conf/log4j.properties");
  }

  public void setConf() {
    // Feature number of train data
    int featureNum = (int) Math.pow(2, 25);

    // Set local deploy mode
    conf.set(AngelConf.ANGEL_DEPLOY_MODE, "LOCAL");

    // Set basic configuration keys
    conf.setBoolean("mapred.mapper.new-api", true);
    conf.set(AngelConf.ANGEL_INPUTFORMAT_CLASS, CombineTextInputFormat.class.getName());
    conf.setBoolean(AngelConf.ANGEL_JOB_OUTPUT_PATH_DELETEONEXIST, true);

    //set angel resource parameters #worker, #task, #PS
    conf.setInt(AngelConf.ANGEL_WORKERGROUP_NUMBER, 1);
    conf.setInt(AngelConf.ANGEL_WORKER_TASK_NUMBER, 1);
    conf.setInt(AngelConf.ANGEL_PS_NUMBER, 1);

    //set sgd LR algorithm parameters #feature #epoch
    conf.set(MLConf.ML_FEATURE_NUM(), String.valueOf(featureNum));
  }

  public void execute() throws Exception {
    setConf();
    String inputPath = "/media/qfeng/软件/downloads/learn_by_angel/data/*.gz";
    String LOCAL_FS = LocalFileSystem.DEFAULT_FS;
    String resultPath = "/media/qfeng/软件/downloads/learn_by_angel/results";
    String savePath = LOCAL_FS + resultPath + "/model";
    String logPath = LOCAL_FS + resultPath + "/log";

    // Set trainning data path
    conf.set(AngelConf.ANGEL_TRAIN_DATA_PATH, inputPath);
    // Set save model path
    conf.set(AngelConf.ANGEL_SAVE_MODEL_PATH, savePath);
    // Set log path
    conf.set(AngelConf.ANGEL_LOG_PATH, logPath);
    // Set actionType train
    conf.set(AngelConf.ANGEL_ACTION_TYPE, MLConf.ANGEL_ML_TRAIN());

    FeaCountRunner runner = new FeaCountRunner();
    runner.train(conf);
  }

  public static void main(String[] args) throws Exception {
    System.out.println(System.getProperty("java.class.path"));
    FeaCountLocal featureCounter = new FeaCountLocal();
    featureCounter.execute();

    System.exit(0);
  }
}
