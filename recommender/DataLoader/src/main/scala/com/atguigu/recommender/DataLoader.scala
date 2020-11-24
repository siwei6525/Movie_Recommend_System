package com.atguigu.recommender

import com.mongodb.casbah.commons.MongoDBObject
import com.mongodb.casbah.{MongoClient, MongoClientURI}
import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}


// 定义样例类 for create the schema of mongodb
/**
 * Movie 数据集
 *
 * 260                                         电影ID，mid
 * Star Wars: Episode IV - A New Hope (1977)   电影名称，name
 * Princess Leia is captured and held hostage  详情描述，descri
 * 121 minutes                                 时长，timelong
 * September 21, 2004                          发行时间，issue
 * 1977                                        拍摄时间，shoot
 * English                                     语言，language
 * Action|Adventure|Sci-Fi                     类型，genres
 * Mark Hamill|Harrison Ford|Carrie Fisher..   演员表，actors
 * George Lucas                                导演，directors
 *
 */
case class Movie(mid: Int, name: String, descri: String, timelong: String, issue: String,
                 shoot: String, language: String, genres: String, actors: String, directors: String)

/**
 * Rating数据集
 *
 * 1,31,2.5,1260759144
 */
case class Rating(uid: Int, mid: Int, score: Double, timestamp: Int)

/**
 * Tag数据集
 *
 * 15,1955,dentist,1193435061
 */
case class Tag(uid: Int, mid: Int, tag: String, timestamp: Int)


// 把mongo的配置封装成样例类
/**
 *
 * @param uri MongoDB连接
 * @param db  MongoDB数据库
 */
case class MongoConfig(uri: String, db: String)

object DataLoader {

  // 定义data路径
  //！！！！若你想用本地文件创建rdd，则应在目录前加入“file://”
  val MOVIE_DATA_PATH = "file:///Users/siwei/Desktop/MovieRecommendSystem/recommender/DataLoader/src/main/resources/movies.csv"
  val RATING_DATA_PATH = "file:///Users/siwei/Desktop/MovieRecommendSystem/recommender/DataLoader/src/main/resources/ratings.csv"
  val TAG_DATA_PATH = "file:///Users/siwei/Desktop/MovieRecommendSystem/recommender/DataLoader/src/main/resources/tags.csv"

  // 定义mongodb的表名
  val MONGODB_MOVIE_COLLECTION = "Movie"
  val MONGODB_RATING_COLLECTION = "Rating"
  val MONGODB_TAG_COLLECTION = "Tag"
  val ES_MOVIE_INDEX = "Movie"

  def main(args: Array[String]): Unit = {

    val config = Map(
      "spark.cores" -> "local[*]", // * 是为了其本地多线程
      "mongo.uri" -> "mongodb://localhost:27017/recommender",
      "mongo.db" -> "recommender",
      "es.httpHosts" -> "localhost:9200",
      "es.transportHosts" -> "localhost:9300",
      "es.index" -> "recommender",
      "es.cluster.name" -> "elasticsearch"
    )

    // 创建一个sparkConf
    val sparkConf = new SparkConf().setMaster(config("spark.cores")).setAppName("DataLoader")
    // 创建一个SparkSession
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    // spark.implicits._   是为了使用 加载数据时转DF的方法--toDF(), _指引用所有
    import spark.implicits._

    // 加载数据 spark.sparkContext.textFile(path) function is used to load the data
    // RDD: Resilient Distributed Dataset is the abstract object of data (like a list of obects) for spark
    val movieRDD = spark.sparkContext.textFile(MOVIE_DATA_PATH)
    // DF: DataFram is the abstract object of data (like a chart) for spark sql
    val movieDF = movieRDD.map(
      item => {
        val attr = item.split("\\^") // 定义切割符 \\做转译，因为是正则，所以 '\'先转译'\', '\'再转译'^'
        // 包成 Movie的case class
        Movie(attr(0).toInt, attr(1).trim, attr(2).trim, attr(3).trim, attr(4).trim, attr(5).trim, attr(6).trim, attr(7).trim, attr(8).trim, attr(9).trim)
      }
    ).toDF()

    val ratingRDD = spark.sparkContext.textFile(RATING_DATA_PATH)

    val ratingDF = ratingRDD.map(item => {
      val attr = item.split(",")
      Rating(attr(0).toInt, attr(1).toInt, attr(2).toDouble, attr(3).toInt)
    }).toDF()

    val tagRDD = spark.sparkContext.textFile(TAG_DATA_PATH)
    //将tagRDD装换为DataFrame
    val tagDF = tagRDD.map(item => {
      val attr = item.split(",")
      Tag(attr(0).toInt, attr(1).toInt, attr(2).trim, attr(3).toInt)
    }).toDF()

    // 隐式定义
    implicit val mongoConfig = MongoConfig(config("mongo.uri"), config("mongo.db"))

    // 将数据保存到MongoDB
    storeDataInMongoDB(movieDF, ratingDF, tagDF)


    spark.stop()
  }

  def storeDataInMongoDB(movieDF: DataFrame, ratingDF: DataFrame, tagDF: DataFrame)(implicit mongoConfig: MongoConfig): Unit = {
    // 新建一个mongodb的连接
    val mongoClient = MongoClient(MongoClientURI(mongoConfig.uri))

    // 如果mongodb中已经有相应的数据库，先删除
    mongoClient(mongoConfig.db)(MONGODB_MOVIE_COLLECTION).dropCollection()
    mongoClient(mongoConfig.db)(MONGODB_RATING_COLLECTION).dropCollection()
    mongoClient(mongoConfig.db)(MONGODB_TAG_COLLECTION).dropCollection()

    // 将DF数据写入对应的mongodb表中
    movieDF.write
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_MOVIE_COLLECTION)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    ratingDF.write
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_RATING_COLLECTION)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    tagDF.write
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_TAG_COLLECTION)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    //对数据表建索引
    mongoClient(mongoConfig.db)(MONGODB_MOVIE_COLLECTION).createIndex(MongoDBObject("mid" -> 1))
    mongoClient(mongoConfig.db)(MONGODB_RATING_COLLECTION).createIndex(MongoDBObject("uid" -> 1))
    mongoClient(mongoConfig.db)(MONGODB_RATING_COLLECTION).createIndex(MongoDBObject("mid" -> 1))
    mongoClient(mongoConfig.db)(MONGODB_TAG_COLLECTION).createIndex(MongoDBObject("uid" -> 1))
    mongoClient(mongoConfig.db)(MONGODB_TAG_COLLECTION).createIndex(MongoDBObject("mid" -> 1))

    mongoClient.close()

  }
}
