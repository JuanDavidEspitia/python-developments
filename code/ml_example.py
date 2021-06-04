import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import Row
from pyspark.sql import SQLContext

sc =SparkContext()
sqlContext = SQLContext(sc)
url = "https://raw.githubusercontent.com/guru99-edu/R-Programming/master/adult_data.csv"
from pyspark import SparkFiles
sc.addFile(url)
sqlContext = SQLContext(sc)


df = sqlContext.read.csv(SparkFiles.get("adult_data.csv"), header=True, inferSchema= True)
df.printSchema()
df.show(5, truncate = False)

from pyspark.sql.types import *

# Write a custom function to convert the data type of DataFrame columns
def convertColumn(df, names, newType):
    for name in names:
        df = df.withColumn(name, df[name].cast(newType))
    return df
# List of continuous features
CONTI_FEATURES  = ['age', 'fnlwgt','capital-gain', 'educational-num', 'capital-loss', 'hours-per-week']
# Convert the type
df_string = convertColumn(df, CONTI_FEATURES, FloatType())
# Check the dataset
df_string.printSchema()


df.select('age','fnlwgt').show(5)

df.groupBy("education").count().sort("count",ascending=True).show()

df.describe().show()

df.describe('capital-gain').show()

# df.drop('education_num').columns

countFilter = df.filter(df.age > 40).count()
print(countFilter)

# TUTORIAL: https://www.guru99.com/pyspark-tutorial.html