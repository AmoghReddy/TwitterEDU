from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from pyspark.context import SparkContext

sc = SparkContext('local', 'test')
# Load and parse the data
def parsePoint(line):
    arr = line.split(',')
    values = [float(x) for x in arr]
    return LabeledPoint(values[0], values[4:])

# data = sc.textFile("lpsa.data")
data = sc.textFile("data/trainFeatLabs.csv")
parsedData = data.map(parsePoint)

# Build the model
model = LinearRegressionWithSGD.train(parsedData, iterations=100, step=0.1)

# Evaluate the model on training data
valuesAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
MSE = valuesAndPreds \
    .map(lambda vp: (vp[0] - vp[1])**2) \
    .reduce(lambda x, y: x + y) / valuesAndPreds.count()
print("Mean Squared Error = " + str(MSE))

# Save and load model
# model.save(sc, "pythonLinearRegressionWithSGDModel")
# sameModel = LinearRegressionModel.load(sc, "pythonLinearRegressionWithSGDModel")