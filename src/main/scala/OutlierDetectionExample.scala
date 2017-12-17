import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.streaming.{Seconds, StreamingContext}
import DetectNUpdate.AnomDetectRMS

/**
  * This outlier detection program implements algorithms, presented in the paper
  * "Streaming Anomaly Detection Using Randomized Matrix Sketching"
  * by Hao Huang and Shiva Prasad Kasiviswanathan,
  * inside the spark streaming framework. Taking data from 2 HDFS folders
  * one for the training and one for the main stream
  *
  * args[0] = Train folder inside the HDFS ex.(/input/Train)
  *
  * args[1] = Monitor folder inside the HDFS ex.(/input/Monitor)
  *
  * args[2] = Normal point output folder inside the HDFS ex.(/output/NormalPoints/)
  *
  * args[3] = Outlier output folder inside the HDFS ex.(/output/Outliers/)
  *
  * args[4] = Batch Duration ex.(1)
  *
  * args[5] = Outlier threshold from 0 to 1 ex.(0.333)
  *
  * args[6] = Thresholding method T for just threshold P based on percentage ex.(T)
  *
  * args[7] = Singular value update method G for global D for deterministic R for randomized (all local to the driver node)
  *           Distr for Spark's distributed SVD
  *
  * args[8] = HDFS IP and port ex.(hdfs://192.168.1.3:9000)
  *
  * args[9] = P for printing the outliers with their score
  * or S to store the outliers and normal points inside the hdfs with their score ex.(S)
  *
  * @author Kyriakos Psarakis (kpsarakis94@gmail.com)
  */

object OutlierDetectionExample {
  def main(args: Array[String]) {

    // args example
   /* val args = new Array[String](10)

    args(0) = "/input/Train"
    args(1) = "/input/Test"
    args(2) = "/output/NormalPoints/"
    args(3) = "/output/Outliers/"
    args(4) = "1"
    args(5) = "0.333"
    args(6) = "P"
    args(7) = "Dist"
    args(8) = "hdfs://192.168.1.3:9000"
    args(9) = "S"
    */

    // check if the user inserted all the args
    if (args.length != 10) {
      System.err.println(
        "Usage: OutlierDetectionExample " +
          "<trainingDir> <testDir> <NPOutDir> <AnomOutDir> <batchDuration> <Threshold> <Method> <UpdateFunc> <hdfs://IP:PORT>")
      System.exit(1)
    }

    // Create the spark conf
    val conf = new SparkConf().setAppName("OutlierDetectionExample")
      // Some useful configurations
      //.setMaster("local[*]")
      //.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      //.set("spark.kryoserializer.buffer.max", "512m")
      //.set("spark.default.parallelism","8")
      //.set("spark.streaming.concurrentJobs","2")
      //.set("spark.streaming.unpersist","true")
      //.set("spark.streaming.backpressure.enabled","true")
      //.set("spark.streaming.blockInterval", "200ms")

    // Create a streaming context
    val ssc = new StreamingContext(conf, Seconds(args(4).toLong))

    // Make spark display only Warnings and Errors
    ssc.sparkContext.setLogLevel("WARN")

    // Create the training DStream
    val trainingData = ssc.textFileStream(args(8)+args(0))
                                            .map(s => Vectors.dense(s.trim.split(" ").map(_.toDouble)))

    // Create the DStream that need to be monitored
    val inputData = ssc.textFileStream(args(8)+args(1))
                                        .map(s => Vectors.dense(s.trim.split(" ").map(_.toDouble)))

    // Initialize the model
    val model = AnomDetectRMS()
      .setSC(ssc.sparkContext)
      .setThreshold(args(5).toDouble)
      .setMethod(args(6))
      .setUpdateFunc(args(7))
      .setHDFS(args(8))
      .setOutputDirs(args(2),args(3))
      .setPrintOrSave(args(9))

    // Train the model
    model.TrainAndUpdate(trainingData)
    // Detect outliers using the model
    model.Detect(inputData)

    //Start the program
    ssc.start()
    ssc.awaitTermination()

  }

}
