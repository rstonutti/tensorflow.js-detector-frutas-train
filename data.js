const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");

const TRAIN_IMAGES_DIR = "./data/train";
const TEST_IMAGES_DIR = "./data/test";

//Esta funcion lee los archivos y devuelve un arreglo con imagenes y etiquetas.
function loadImages(dataDir) {
  const images = [];
  const labels = [];

  //Lee el contenido del directorio y devuelve un array con todos los objetos dentro.
  var files = fs.readdirSync(dataDir);
  for (let i = 0; i < files.length; i++) {
    if (!files[i].toLocaleLowerCase().endsWith(".png")) {
      continue;
    }

    var filePath = path.join(dataDir, files[i]);
    var buffer = fs.readFileSync(filePath);

    var imageTensor = tf.node
      .decodeImage(buffer)
      .resizeNearestNeighbor([96, 96]) //Cambiamos el tamaño de las imagenes para que todas tengan el mismo tamaño
      .toFloat() //Todos los valores se convierten en flotantes
      .div(tf.scalar(255.0))
      .expandDims();
    images.push(imageTensor);

    /* las imagenes deben cumplir con el formato n_fruta.png */
    var banana = files[i].toLocaleLowerCase().endsWith("banana.png");
    var manzana = files[i].toLocaleLowerCase().endsWith("manzana.png");
    var naranja = files[i].toLocaleLowerCase().endsWith("naranja.png");
    var pera = files[i].toLocaleLowerCase().endsWith("pera.png");
    var uva = files[i].toLocaleLowerCase().endsWith("uva.png");

    if (banana == true) {
      labels.push(1);
    } else if (manzana == true) {
      labels.push(2);
    } else if (naranja == true) {
      labels.push(3);
    } else if (pera == true) {
      labels.push(4);
    } else if (uva == true) {
      labels.push(0);
    }
  }
  console.log("Las etiquetas son");
  console.log(labels);
  return [images, labels];
}

/* Clase para manejar el entrenamiento y testeo de datos. */
class FruitDataset {
  constructor() {
    this.trainData = [];
    this.testData = [];
  }

  /** Cargamos los datos de entrenamiento y testeo. */
  loadData() {
    console.log("Cargando imagenes...");
    this.trainData = loadImages(TRAIN_IMAGES_DIR);
    this.testData = loadImages(TEST_IMAGES_DIR);
    console.log("Imagenes cargadas satisfactoriamente.");
  }
  //Devolvemos los datos cargados
  getTrainData() {
    return {
      images: tf.concat(this.trainData[0]),
      labels: tf
        .oneHot(tf.tensor1d(this.trainData[1], "int32"), 5)
        .toFloat() /* colocamos 5 porque utilizamos 5 frutas */,
    };
  }

  getTestData() {
    return {
      images: tf.concat(this.testData[0]),
      labels: tf.oneHot(tf.tensor1d(this.testData[1], "int32"), 5).toFloat(),
    };
  }
}

module.exports = new FruitDataset();
console.log("Todos listo.");
