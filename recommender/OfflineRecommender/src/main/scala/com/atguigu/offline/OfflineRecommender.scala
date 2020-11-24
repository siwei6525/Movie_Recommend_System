package com.atguigu.offline

import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.sql.SparkSession

// 基于隐语义（LFM(latent factor model)）模型的离线推荐：
// data用的是评分，算法用的是ALS（交替最小二乘法）（Alternating Least Squares）

// 基于评分数据的LFM，只需要rating数据（评分）
// 这里这个电影评分的样例类叫MovieRating，是为了区分之前的Rating样例类
case class MovieRating(uid: Int, mid: Int, score: Double, timestamp: Int )

case class MongoConfig(uri:String, db:String)

// 定义一个基准推荐对象
case class Recommendation( mid: Int, score: Double )

// 定义基于  预测评分  的用户推荐列表 （给每个用户的推荐）
case class UserRecs( uid: Int, recs: Seq[Recommendation] )

// 定义基于  LFM电影特征向量  的电影相似度列表  （对每个电影的推荐，即电影相似度）
case class MovieRecs( mid: Int, recs: Seq[Recommendation] )

object OfflineRecommender {

  // 定义表名和常量

  // 这个是原来的data，我们要使用的
  val MONGODB_RATING_COLLECTION = "Rating"

  // 这两个是新生成的结果
  val USER_RECS = "UserRecs"
  val MOVIE_RECS = "MovieRecs"

  // 推荐数目
  val USER_MAX_RECOMMENDATION = 20

  def main(args: Array[String]): Unit = {
    val config = Map(
      "spark.cores" -> "local[*]",
      "mongo.uri" -> "mongodb://localhost:27017/recommender",
      "mongo.db" -> "recommender"
    )

    // 创建spark配置
    val sparkConf = new SparkConf().setMaster(config("spark.cores")).setAppName("OfflineRecommender")

    // 创建一个SparkSession
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    import spark.implicits._

    implicit val mongoConfig = MongoConfig(config("mongo.uri"), config("mongo.db"))


    // 加载数据
    val ratingRDD = spark.read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_RATING_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[MovieRating]
      .rdd
      .map( rating => ( rating.uid, rating.mid, rating.score ) )    // 转化成rdd，并且去掉时间戳
      .cache()  // rdd持久化在缓存里 为了提升算笛卡尔积的效率

    // 从rating数据中提取所有的uid和mid，并去重
    val userRDD = ratingRDD.map(_._1).distinct()
    val movieRDD = ratingRDD.map(_._2).distinct()


    // 核心代码
    // 训练隐语义模型
    val trainData = ratingRDD.map( x => Rating(x._1, x._2, x._3) )

    val (rank, iterations, lambda) = (300, 5, 0.1)

    // 得到模型
    val model = ALS.train(trainData, rank, iterations, lambda)



    // 基于用户和电影的隐特征（评分），计算预测评分，得到用户的推荐列表
    // 1，首先需要一个空的矩阵 记录结果userMovies
    // 计算user和movie的笛卡尔积（cartesian），得到一个空评分矩阵
    val userMovies = userRDD.cartesian(movieRDD)

    // 2，调用model的predict方法预测评分
    val preRatings = model.predict(userMovies)

    // 3，填入预测评分
    val userRecs = preRatings
      .filter(_.rating > 0)    // 过滤出评分大于0的项
      .map(rating => ( rating.user, (rating.product, rating.rating) ) )
      .groupByKey()
      .map{
        case (uid, recs) => UserRecs( uid, recs.toList.sortWith(_._2>_._2).take(USER_MAX_RECOMMENDATION).map(x=>Recommendation(x._1, x._2)) )
      }
      .toDF()

    userRecs.write
      .option("uri", mongoConfig.uri)
      .option("collection", USER_RECS)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    spark.stop()
  }
}
