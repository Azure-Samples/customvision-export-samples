using Microsoft.ML;
using Microsoft.ML.Data;
using System.CommandLine;

// Command Line Config
var imagePathOption = new Option<string>(name:"--image_path",description:"The path of the image to run inference on,");
var modelPathOption = new Option<string>(name:"--model_path", description:"The path of the ONNX model used for inferencing.");
var labelPathOption = new Option<string>(name:"--labels_path",description: "The path of the labels file for your ONNX model.");
var confidenceOption = new Option<float>(name:"--confidence", description: "Value used to filter out bounding boxes with lower confidence.", getDefaultValue: () => 0.7f);

var rootCommand = new RootCommand("Sample application to run inferencing using an ML.NET pipeline and an Azure Custom Vision ONNX model");

rootCommand.AddOption(imagePathOption);
rootCommand.AddOption(modelPathOption);
rootCommand.AddOption(labelPathOption);
rootCommand.AddOption(confidenceOption);

var CLIHandler = (string image, string model, string labels, float confidence) => 
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

    RunInference(image!, model!, labels!,confidence!); 
};

rootCommand.SetHandler(CLIHandler, imagePathOption, modelPathOption, labelPathOption,confidenceOption);

await rootCommand.InvokeAsync(args);

// Run Inference Helper Function
static void RunInference(string imagePath, string modelPath, string labelPath,float confidence)
{
    // Initialize MLContext
    var ctx = new MLContext();

    // Load labels
    var labels = File.ReadAllLines(labelPath);

    // Define inferencing pipeline
    var pipeline = 
        ctx.Transforms.LoadImages(outputColumnName: "Image",null,inputColumnName:"ImagePath")
            .Append(ctx.Transforms.ResizeImages(outputColumnName: "ResizedImage", imageWidth: 320, imageHeight: 320, inputColumnName: "Image", resizing: Microsoft.ML.Transforms.Image.ImageResizingEstimator.ResizingKind.Fill))
            .Append(ctx.Transforms.ExtractPixels(outputColumnName: "Pixels", inputColumnName: "ResizedImage", offsetImage:255, scaleImage: 1))
            .Append(ctx.Transforms.CopyColumns(outputColumnName:"image_tensor","Pixels"))
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

    // Get bounding boxes
    var boundingBoxes = prediction.ToBoundingBoxes(labels, MLImage.CreateFromFile(input.ImagePath));

    // Get top bounding boxes based on probability
    var topBoundingBoxes = 
        boundingBoxes
            .Where(x => x.Probability > confidence)
            .OrderByDescending(x => x.Probability)
            .ToArray();

    // Print out bounding box information to the console
    foreach(var b in topBoundingBoxes)
    {
        Console.WriteLine(b);
    }    
}

class ModelInput
{
    public string? ImagePath { get; set; }
}

class ModelOutput
{

    [ColumnName("detected_boxes")]
    [VectorType()]
    public float[]? Boxes {get;set;}

    [ColumnName("detected_scores")]
    [VectorType()]
    public float[]? Scores {get;set;}

    [ColumnName("detected_classes")]
    [VectorType()]
    public long[]? Classes {get;set;}

    // Helper functions

    public BoundingBox[] ToBoundingBoxes(string[] labels, MLImage originalImage)
    {
        var bboxCoordinates = 
            this.Boxes!
                .Chunk(4)
                .ToArray();

        var boundingBoxes = 
            bboxCoordinates
                .Select((coordinates,idx) => 
                    new BoundingBox
                    {
                        TopLeft=(X: coordinates[0] * originalImage.Width, Y: coordinates[1] * originalImage.Height),
                        BottomRight=(X: coordinates[2] * originalImage.Width, Y: coordinates[3] * originalImage.Height),
                        PredictedClass=labels[this.Classes![idx]],
                        Probability=this.Scores![idx]
                    })
                .ToArray();

        return boundingBoxes;
    }
}

public class BoundingBox
{
    public (float X, float Y) TopLeft {get;set;}
    public (float X, float Y) BottomRight {get;set;}
    public string? PredictedClass {get;set;}
    public float Probability {get;set;}

    public override string ToString() => 
        $"Top Left (x,y): ({TopLeft.X},{TopLeft.Y})\nBottom Right (x,y): ({BottomRight.X},{BottomRight.Y})\nClass: {PredictedClass}\nProbability: {Probability})";
}
