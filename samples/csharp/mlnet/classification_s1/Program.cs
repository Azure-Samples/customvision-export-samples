using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.Transforms.Image.ImagePixelExtractingEstimator;
using System.CommandLine;

// Command Line Config
var imagePathOption = new Option<string>(name:"--image_path",description:"The path of the image to run inference on,");
var modelPathOption = new Option<string>(name:"--model_path", description:"The path of the ONNX model used for inferencing.");
var labelPathOption = new Option<string>(name:"--labels_path",description: "The path of the labels file for your ONNX model.");

var rootCommand = new RootCommand("Sample application to run inferencing using an ML.NET pipeline and an Azure Custom Vision ONNX model");

rootCommand.AddOption(imagePathOption);
rootCommand.AddOption(modelPathOption);
rootCommand.AddOption(labelPathOption);

var CLIHandler = (string image, string model, string labels) => 
{

    if(image == null)
    {
        Console.WriteLine("Missing --image_path parameter");
        return;
    }
    
    if(model == null)
    {
        Console.WriteLine("Missing --model_path parameter");
        return;
    }

    if(labels == null)
    {
        Console.WriteLine("Missing --labels_path parameter");
        return;
    }

    RunInference(image!, model!, labels!); 
};

rootCommand.SetHandler(CLIHandler, imagePathOption, modelPathOption, labelPathOption);

await rootCommand.InvokeAsync(args);

static void RunInference(string imagePath, string modelPath, string labelPath)
{
    // Initialize MLContext
    var ctx = new MLContext();

    // Load labels
    var labels = File.ReadAllLines(labelPath);

    // Define inferencing pipeline
    var pipeline = 
        ctx.Transforms.LoadImages(outputColumnName: "Image",null,inputColumnName:"ImagePath")
            .Append(ctx.Transforms.ResizeImages(outputColumnName: "ResizedImage", imageWidth: 300, imageHeight: 300, inputColumnName: "Image", resizing: Microsoft.ML.Transforms.Image.ImageResizingEstimator.ResizingKind.Fill))
            .Append(ctx.Transforms.ExtractPixels(outputColumnName: "Pixels", inputColumnName: "ResizedImage", offsetImage:255, scaleImage: 1, orderOfExtraction: ColorsOrder.ABGR))
            .Append(ctx.Transforms.CopyColumns(outputColumnName:"data", inputColumnName: "Pixels"))
            .Append(ctx.Transforms.ApplyOnnxModel(modelFile: modelPath));

    // Define empty DataView to create inferencing pipeline
    var emptyDv = ctx.Data.LoadFromEnumerable(new ModelInput[] {});

    // Build inferencing pipeline
    var model = pipeline.Fit(emptyDv);

    // (Optional)
    ctx.Model.Save(model,emptyDv.Schema,"model.mlnet");

    // Use inferencing pipeline
    var input = new ModelInput {ImagePath=imagePath};
    var predictionEngine = ctx.Model.CreatePredictionEngine<ModelInput,ModelOutput>(model);
    var prediction = predictionEngine.Predict(input);
    var predictedLabel = prediction.GetPredictedLabel();

    Console.WriteLine($"Image {imagePath} classified as {labels[predictedLabel]}");
}

class ModelInput
{
    public string ImagePath { get; set; }
}

class ModelOutput
{
    [ColumnName("model_output")]
    public float[] Scores { get; set; }

    public int GetPredictedLabel() => 
        Array.IndexOf(this.Scores, this.Scores.Max());
}