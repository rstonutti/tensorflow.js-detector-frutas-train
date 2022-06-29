const tf = require("@tensorflow/tfjs-node");

const data = require("./data");
const model = require("./model");

async function run(epochs, batchSize, modelSavePath) {
  data.loadData();

  const { images: trainImages, labels: trainLabels } = data.getTrainData();
  console.log("Imagenes de entrenamiento (Shape): " + trainImages.shape);
  console.log("Etiquetas de entrenamiento (Shape): " + trainLabels.shape);

  //Imprimimos en pantalla un resumen del modelo
  model.summary();

  const validationSplit = 0.15;
  await model.fit(trainImages, trainLabels, {
    epochs,
    batchSize,
    validationSplit,
  });

  const { images: testImages, labels: testLabels } = data.getTestData();
  const evalOutput = model.evaluate(testImages, testLabels);

  console.log(
    `\nResultados:\n` +
      `Perdida = ${evalOutput[0].dataSync()[0].toFixed(3)}; ` +
      `Precisión = ${evalOutput[1].dataSync()[0].toFixed(3)}`
  );

  if (modelSavePath != null) {
    await model.save(`file://${modelSavePath}`);
    console.log(`Modelo guardado en la ruta: ${modelSavePath}`);
  }
}

run(10, 4, "./model");
