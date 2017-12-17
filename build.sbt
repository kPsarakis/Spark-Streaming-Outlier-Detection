name := "SparkStreamingOutlierDetection"

version := "0.1"

scalaVersion := "2.11.8"

//For IDE local mode
/*
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.2.0"
libraryDependencies += "org.apache.spark" %% "spark-core" % "2.2.0"
libraryDependencies += "org.apache.spark" %% "spark-streaming" % "2.2.0"
*/

//For sbt assembly
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.2.0"% "provided" withSources()
libraryDependencies += "org.apache.spark" %% "spark-core" % "2.2.0"% "provided" withSources()
libraryDependencies += "org.apache.spark" %% "spark-streaming" % "2.2.0"% "provided" withSources()


        