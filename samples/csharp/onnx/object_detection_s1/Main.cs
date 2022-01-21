using System.CommandLine;
using System.IO;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;


namespace CustomVision
{
    class Program
    {
        static async Task Main(string[] args)
        {
            var modelFilepathArgument = new Argument<FileInfo>("model_filepath");
            var imageFilepathArgument = new Argument<FileInfo>("image_filepath");
            var command = new RootCommand
            {
                modelFilepathArgument,
                imageFilepathArgument
            };

            command.SetHandler((FileInfo modelFilepath, FileInfo imageFilepath) => {
                Predict(modelFilepath, imageFilepath);
            }, modelFilepathArgument, imageFilepathArgument);

            await command.InvokeAsync(args);
        }

        static void Predict(FileInfo modelFilepath, FileInfo imageFilepath)
        {
            var session = new InferenceSession(modelFilepath.ToString());

            Tensor<float> tensor = LoadInputTensor(imageFilepath, 320);
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor<float>("image_tensor", tensor)
            };

            var resultsCollection = session.Run(inputs);
            var resultsDict = resultsCollection.ToDictionary(x => x.Name, x => x);
            var detectedBoxes = resultsDict["detected_boxes"].AsTensor<float>();
            var detectedClasses = resultsDict["detected_classes"].AsTensor<long>();
            var detectedScores = resultsDict["detected_scores"].AsTensor<float>();

            var numBoxes = detectedClasses.Length;

            for (var i = 0; i < numBoxes; i++) {
                var score = detectedScores[0, i];
                var classId = detectedClasses[0, i];
                var x = detectedBoxes[0, i, 0];
                var y = detectedBoxes[0, i, 1];
                var x2 = detectedBoxes[0, i, 2];
                var y2 = detectedBoxes[0, i, 3];
                Console.WriteLine("Label: {0}, Probability: {1}, Box: ({2}, {3}) ({4}, {5})", classId, score, x, y, x2, y2);
            }
        }

        // Load an image file and create a RGB[0-255] tensor.
        static Tensor<float> LoadInputTensor(FileInfo imageFilepath, int imageSize)
        {
            var input = new DenseTensor<float>(new[] {1, 3, imageSize, imageSize});
            using (var image = Image.Load<Rgb24>(imageFilepath.ToString()))
            {
                image.Mutate(x => x.Resize(imageSize, imageSize));

                for (int y = 0; y < image.Height; y++)
                {
                    Span<Rgb24> pixelSpan = image.GetPixelRowSpan(y);
                    for (int x = 0; x < image.Width; x++)
                    {
                        input[0, 0, y, x] = pixelSpan[x].R;
                        input[0, 1, y, x] = pixelSpan[x].G;
                        input[0, 2, y, x] = pixelSpan[x].B;
                    }
                }
            }
            return input;
        }
    }
}