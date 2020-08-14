const fs = require('fs')
const argv = require('yargs').usage('Usage: $0 <image_filepath>').demandCommand(1).argv;
const cvstfjs = require('@microsoft/customvision-tfjs-node')

async function run(image_filepath) {
	const model = new cvstfjs.ObjectDetectionModel();
	await model.loadModelAsync('file://model.json')

  fs.readFile(image_filepath, async function (err, data) {
    if (err) {
      throw err;
    }

    const result = await model.executeAsync(data);
    console.log(result);
  });
}

run(argv._[0])
