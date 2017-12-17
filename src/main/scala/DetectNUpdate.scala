//JAVA
import java.net.URI

import org.apache.spark.mllib.linalg.SparseVector

//BREEZE
import breeze.linalg.eig.DenseEig
import breeze.linalg.svd.DenseSVD
import breeze.linalg.{argsort, diag, eig, qr, svd, DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics.{pow, sqrt}
import breeze.stats.distributions.Gaussian

//HADOOP
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FSDataOutputStream, FileSystem, Path}

//SPARK CORE
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.broadcast.Broadcast

//SPARK STREAMING
import org.apache.spark.streaming.dstream.DStream

//SPARK MLLIB
import org.apache.spark.mllib.feature.Normalizer
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector, Matrices, Matrix, SingularValueDecomposition, Vector, Vectors}


object DetectNUpdate {

  case class AnomDetectRMS() extends Serializable {

    // Rand and Deterministic sketches
    var Et, Bt: BDM[Double] = _

    // Im - Uk*Uk'
    var Uk: Array[Double] = _

    //Global Update variables
    var Ut: BDM[Double] = _
    var St: BDV[Double] = _

    // Configuration variables
    var thresholdValue: Double = 0
    var method, UpdateFunc: String = ""

    // L2 Normalizer
    var normalizerL2: Normalizer = new Normalizer()

    //RUNTIME MEASUREMENTS
    var t1: Long = 0
    var duration: Double = 0
    var avgDuration: Double = 0

    //HDFS CONFIGURATIONS
    var HdfsIpPort, NormalPointDir, OutlierDir, PoS: String = ""
    var hdfs: FileSystem = _
    var sc: SparkContext = _
    var OutlierPath, NormalPointsPath: Path = _
    var rddCounter: Int = 0

    //Distributed deterministic sketch
    var Btd: RDD[Vector] = _

    //Feature size, rank, sketch size
    var m, k, l: Int = 0

    // set the Spark Context
    def setSC(s: SparkContext): this.type = {
      sc = s
      this

    }

    def setPrintOrSave(pos: String): this.type = {
      require(pos.equals("P") || pos.equals("S"), s"P (Print Outliers on terminal) or S (Save Outliers and normal points to HDFS) $pos")
      PoS = pos
      this
    }

    // set the Threshold value
    def setThreshold(t: Double): this.type = {
      require(t >= 0, s"Threshold must be positive but got $t")
      thresholdValue = t
      this
    }

    // set the Thresholding method
    def setMethod(me: String): this.type = {
      require(me.equals("T") || me.equals("P"), s"Method must be T (Threshold) or P (Percentage) $me")
      method = me
      this
    }

    // set the Update Function
    def setUpdateFunc(u: String): this.type = {
      require(u.equals("R") || u.equals("G") || u.equals("D") || u.equals("Dist"), s"Method must be R (Random) G (Global) or D (Det)  $u")
      UpdateFunc = u
      this
    }

    // set the HDFS
    def setHDFS(hip: String): this.type = {
      HdfsIpPort = hip
      hdfs = FileSystem.get(URI.create(hip), new Configuration())
      this
    }

    // set the HDFS paths
    def setOutputDirs(normal: String, Out: String): this.type = {
      NormalPointDir = normal
      OutlierDir = Out
      this
    }

    // Train with the train DStream
    def TrainAndUpdate(data: DStream[Vector]) {

      data.foreachRDD { (rdd: RDD[Vector], _) =>

        // do not allow empty rdd
        if (!rdd.isEmpty()) {

          //feature size
          m = rdd.take(1).toVector(0).size

          // thought to be good for the specific feature size
          k = m / 5 // rank
          l = math.sqrt(m).toInt // sketch size

          // k <= l so if l<k make k = l
          if (k > l)
            k = l

          //Train with the specific Update Function
          UpdateFunc match {
            case "R" => RandUpdate(normalizerL2.transform(rdd))
            case "G" => GlobalUpdate(normalizerL2.transform(rdd))
            case "Dist" => DetUpdateDistr(normalizerL2.transform(rdd))
            case "D" => DetUpdate(normalizerL2.transform(rdd))
          }

        }
      }
    }

    // Detect outliers inside the specified DStream
    def Detect(data: DStream[Vector]) {

      data.foreachRDD { (rdd: RDD[Vector], _) =>

        // do not allow empty rdd
        if (!rdd.isEmpty()) {

          // check if the model is trained
          assertInitialized()

          // for the hdfs writes and average runtime
          rddCounter += 1

          // count runtime start
          t1 = System.nanoTime

          //Detect with the specific Update Function
          UpdateFunc match {
            case "R" => RandUpdate(AnomDetect(rdd))
            case "G" => GlobalUpdate(AnomDetect(rdd))
            case "Dist" => DetUpdateDistr(AnomDetect(rdd))
            case "D" => DetUpdate(AnomDetect(rdd))
          }

          // runtime duration
          duration = (System.nanoTime - t1) / 1e9d
          println("DETECT DURATION : " + duration + " (sec)")

          // average runtime
          avgDuration += duration
          println("AVERAGE DETECT DURATION : " + avgDuration / rddCounter)

        }
      }
    }

    // Custom transformation of an RDD[Vector] output = Normal points
    def AnomDetect(Yt: RDD[Vector]): RDD[Vector] = {

      var NormalPoints, Outliers: RDD[(Vector, Double)] = null



      // Broadcast the variables to all the workers
      val bm: Broadcast[Int] = sc.broadcast(m) //feature size
      val bTV: Broadcast[Double] = sc.broadcast(thresholdValue) // threshold
      val bMeth: Broadcast[Char] = sc.broadcast(method(0)) // thresholding method
      val bPoS: Broadcast[Char] = sc.broadcast(PoS(0)) // Print or save the outliers
      val bU = sc.broadcast(Uk)

      // Calculate score and create a KV pair (point (Vector) Key, score (Double) Value)
      val X: RDD[(Vector, Double)] = Yt.zip(normalizerL2.transform(Yt).map(
        v => Vectors.norm(Matrices.dense(bm.value, bm.value, bU.value).multiply(v), 2.0))).cache()

      bMeth.value match {

        //Percentage mode (we know how many outliers to expect in each batch) used only for the ROC experiment
        case 'P' =>

          // Expected Normal Points for P method
          val bXN: Broadcast[Long] = sc.broadcast((Yt.count * (1 - bTV.value)).toLong)

          // Sort the scores in ascending order (sort = costly wide transformation)
          val Xs: RDD[((Vector, Double), Long)] = X.sortBy(_._2, ascending = true).zipWithIndex().cache()

          // Take the lowest scores as  Normal Points
          NormalPoints = Xs.filter(_._2 < bXN.value).map(_._1)

          // Take all the others as  Outliers
          Outliers = Xs.filter(_._2 >= bXN.value).map(_._1)

        //Threshold mode (the primary method suggested on the paper (10x faster <Only filter>))
        case 'T' =>

          NormalPoints = X.filter(_._2 < bTV.value)

          Outliers = X.filter(_._2 >= bTV.value)

      }


      bPoS.value match {

        case 'P' =>

          Outliers.foreach(println)

        case 'S' =>

          //HDFS Output paths
          OutlierPath = new Path(HdfsIpPort + OutlierDir + rddCounter + ".txt")
          NormalPointsPath = new Path(HdfsIpPort + NormalPointDir + rddCounter + ".txt")

          //Ways to save the data to the HDFS

          //One file per partition
          //NormalPoints.saveAsTextFile(NormalPointsPath.toString)
          //Outliers.saveAsTextFile(OutlierPath.toString)

          //Just one file
          saveToHDFS(NormalPoints,NormalPointsPath)
          saveToHDFS(Outliers,OutlierPath)

      }

      normalizerL2.transform(NormalPoints.map(_._1))

    }

    //Randomized Update (computation occurs inside the driver node)
    def RandUpdate(rdd: RDD[Vector]) {

      // If only outliers where detected
      if(!rdd.isEmpty()) {

        //Concatenate sketch with input matrix
        val Mt: BDM[Double] = {
          if (Et == null) {
            BDM(rdd.collect().map(_.toArray): _*).t
          }
          else {
            BDM.horzcat(Et, BDM(rdd.collect().map(_.toArray): _*).t)
          }
        }

        // Calculate here because we use it 2 times
        val MMt: BDM[Double] = Mt * Mt.t

        //QR of a small matrix mx100l
        val Q: BDM[Double] = qr.reduced.justQ(MMt * BDM.rand(m, l * 100, Gaussian(0, 0.1)))

        // EVD of a small matrix mxm
        val es: DenseEig = eig(Q.t * MMt * Q)

        //Order the eigen values from greatest to smallest
        val order: IndexedSeq[Int] = argsort(es.eigenvalues).reverse
        val s: BDV[Double] = es.eigenvalues(order).toDenseVector

        //Approximate U
        val U: BDM[Double] = Q * es.eigenvectors(::, order).toDenseMatrix

        //Create new sketch based on the shrinking step of Frequent Directions
        Et = U(::, 0 until l) * diag(sqrt(s(0 until l) - s(l - 1)))

        // Im - Uk*Uk' (done here due to lack on Matrix operations in MlLib)
        Uk = (BDM.eye[Double](m) - U(::, 0 until k) * U(::, 0 until k).t).data
      }
    }

    //Deterministic Update (computation occurs inside the driver node)
    def DetUpdate(rdd: RDD[Vector]) {

      // If only outliers where detected
      if(!rdd.isEmpty()) {
        //Concatenate sketch with input matrix
        val D: BDM[Double] = {
          if (Bt == null) {
            BDM(rdd.collect().map(_.toArray): _*).t
          }
          else {
            BDM.horzcat(Bt, BDM(rdd.collect().map(_.toArray): _*).t)
          }
        }

        //Low Rank SVD
        val _svd: DenseSVD = svd.reduced(D)

        val Ul: BDM[Double] = _svd.U(::, 0 until l)

        val sl: BDV[Double] = _svd.S(0 until l)

        Bt = Ul * diag(sqrt(pow(sl, 2) - pow(sl(-1), 2)))

        // Im - Uk*Uk' (done here due to lack on Matrix operations in MlLib)
        Uk = (BDM.eye[Double](m) - Ul(::, 0 until k) * Ul(::, 0 until k).t).data

      }
    }

    //Using the mllib framework for Distributed SVD
    // Doesnt work well with few workers
    // reason being it does many wide transformations
    def DetUpdateDistr(Nt: RDD[Vector]) {

      if (Nt.count() > 1) { // This statement is only for unstable HDFS input for all other cases comment

        // Broadcast the variables to all the workers
        val bm: Broadcast[Int] = sc.broadcast(m) //feature size
        val bk: Broadcast[Int] = sc.broadcast(k) // rank
        val bl: Broadcast[Int] = sc.broadcast(l) // sketch size

        //Concatenate sketch with input matrix
        val D: RDD[Vector] = {
          if (Btd == null) Nt else transposeRDD(Btd).union(Nt)
        }.cache()

        // MlLibs Low rank SVD
        val svd: SingularValueDecomposition[RowMatrix, Matrix] = new RowMatrix(transposeRDD(D)).computeSVD(bl.value, computeU = true)

        //Create the sketch of the current time step
        Btd = svd.U.multiply(
          DenseMatrix.diag(
            Vectors.dense(svd.s.toArray
              .map(x => x * x)
              .map(x => x - pow(svd.s.toArray.last, 2))
              .map(x => sqrt(x))))).rows.cache()

        // Uk
        val tmp: BDM[Double] = new BDM(bm.value, bk.value, transposeRDD(svd.U.rows).take(bk.value).flatMap(_.toArray))

        // Im - Uk*Uk' (done here due to lack on Matrix operations in MlLib)
        Uk = (BDM.eye[Double](bm.value) - tmp * tmp.t).data

      }
    }

    //First update method only for testing cause its
    // memory requirements and SVD cost are too big
    // in a real time setting
    def GlobalUpdate(rdd: RDD[Vector]) {

      // If only outliers where detected
      if(!rdd.isEmpty()) {

        var svdC: DenseSVD = null

        if (St == null) {

          svdC = svd(BDM(rdd.collect().map(_.toArray): _*).t)

          Ut = svdC.U
          St = svdC.S

        } else {

          val F = BDM.horzcat(diag(St), Ut.t * BDM(rdd.collect().map(_.toArray): _*).t)

          svdC = svd(F)

          Ut = Ut * svdC.U
          St = svdC.S

        }

        // Im - Uk*Uk' (done here due to lack on Matrix operations in MlLib)
        Uk = (BDM.eye[Double](m) - Ut(::, 0 until k) * Ut(::, 0 until k).t).data
      }
    }

    //Our implementation to save the rdd as one file to the hdfs
    def saveToHDFS(rdd: RDD[(Vector, Double)], p: Path): Unit = {
      val y: FSDataOutputStream = hdfs.create(p, true)
      y.write(rdd.collect().mkString("\n").replaceAll("""[\p{Punct}&&[^.-]]""", " ").getBytes())
      y.close()
    }

    //Check if the model has been trained
    private[this] def assertInitialized(): Unit = {
      if (Uk == null) {
        throw new IllegalStateException(
          "The model has not been trained")
      }
    }

    //Transpose an RDD Vector in a Distributed fashion
    def transposeRDD(X: RDD[Vector]): RDD[Vector] = {

      val n = X.count().toInt

      X.zipWithIndex.flatMap {
        case (sp: SparseVector, i: Long) => sp.indices.zip (sp.values).map {case (j, value) => (i, j, value)}
        case (dp: DenseVector, i: Long) => Range(0, n).toArray.zip(dp.values).map { case (j, value) => (i, j, value) }
      }
        .sortBy(t => t._1)
        .groupBy(t => t._2)
        .map { case (i, g) => val (_, values) = g.map { case (idx, _, value) => (idx.toInt, value) }.unzip
          (i, Vectors.dense(values.toArray))
        }
        .sortBy(t => t._1).map(t => t._2)

    }

  }

}